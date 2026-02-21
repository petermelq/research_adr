"""Train experiment models on Russell residual features with configurable walk-forward windows."""

import argparse
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler


def load_experiment_universe():
    adr_info = pd.read_csv(Path('data/raw/adr_info.csv'))
    adr_info['adr_ticker'] = adr_info['adr'].str.replace(' US Equity', '', regex=False)
    universe = adr_info['adr_ticker'].dropna().unique().tolist()
    return set(universe)


def _normalize_bbg_equity_ticker(raw):
    if pd.isna(raw):
        return None
    t = str(raw).strip()
    if not t:
        return None
    return t.split()[0].upper()


def load_historical_russell_constituents(path):
    df = pd.read_csv(path)
    date_col = 'as_of_date' if 'as_of_date' in df.columns else 'DATE'
    ticker_col = 'ticker' if 'ticker' in df.columns else 'id'
    if date_col not in df.columns or ticker_col not in df.columns:
        raise ValueError(f"historical constituent file missing required columns: {path}")

    out = pd.DataFrame({
        'as_of_date': pd.to_datetime(df[date_col]).dt.normalize(),
        'ticker': df[ticker_col].map(_normalize_bbg_equity_ticker),
    }).dropna()
    out = out.drop_duplicates(['as_of_date', 'ticker']).sort_values(['as_of_date', 'ticker'])

    snapshot_dates = np.array(sorted(out['as_of_date'].unique()), dtype='datetime64[ns]')
    constituent_sets = {
        pd.Timestamp(d): set(out.loc[out['as_of_date'] == d, 'ticker'].tolist())
        for d in pd.to_datetime(snapshot_dates)
    }
    return snapshot_dates, constituent_sets


def load_turnover(path):
    t = pd.read_csv(path, index_col=0, parse_dates=True)
    t.columns = [_normalize_bbg_equity_ticker(c) for c in t.columns]
    t = t.loc[:, [c for c in t.columns if c is not None]]
    t = t[~t.index.duplicated(keep='last')].sort_index()
    return t


def _last_snapshot_on_or_before(snapshot_dates, ts):
    i = np.searchsorted(snapshot_dates, np.datetime64(pd.Timestamp(ts).normalize()), side='right') - 1
    if i < 0:
        return None
    return pd.Timestamp(snapshot_dates[i])


def select_window_feature_columns(
    feature_cols,
    window_dates,
    train_start,
    val_end,
    snapshot_dates,
    constituent_sets,
    turnover_df,
    min_turnover,
):
    start_snapshot = _last_snapshot_on_or_before(snapshot_dates, train_start)
    end_snapshot = _last_snapshot_on_or_before(snapshot_dates, val_end)
    if start_snapshot is None or end_snapshot is None or start_snapshot > end_snapshot:
        return []

    # Constituents present on every monthly snapshot from train-start snapshot through val-end snapshot.
    monthly_window = [
        pd.Timestamp(d)
        for d in pd.to_datetime(snapshot_dates)
        if start_snapshot <= pd.Timestamp(d) <= end_snapshot
    ]
    if not monthly_window:
        return []

    always_in_index = set(constituent_sets.get(monthly_window[0], set()))
    for d in monthly_window[1:]:
        always_in_index &= constituent_sets.get(d, set())

    # Also include names in end-of-validation snapshot that are liquid every day in train+val.
    end_snapshot_set = set(constituent_sets.get(end_snapshot, set()))
    liquid_set = set()
    if len(window_dates) > 0 and not turnover_df.empty:
        end_cols = [c for c in end_snapshot_set if c in turnover_df.columns]
        if end_cols:
            t_window = turnover_df.reindex(window_dates, columns=end_cols)
            is_liquid = (t_window >= min_turnover) & t_window.notna()
            liquid_set = set(is_liquid.columns[is_liquid.all(axis=0)])

    eligible = always_in_index | liquid_set
    return [c for c in feature_cols if c.replace('russell_', '', 1) in eligible]


class LinearModelBundle:
    def __init__(self, feature_names, w_raw, c_raw=0.0):
        self.feature_names = feature_names
        self.w_raw = np.asarray(w_raw, dtype=np.float32)
        self.c_raw = float(c_raw)

    def predict_raw(self, X_raw):
        return X_raw @ self.w_raw + self.c_raw


def compute_ic(predictions, actuals):
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    if mask.sum() < 2:
        return 0.0
    p = predictions[mask]
    a = actuals[mask]
    if np.std(p) == 0 or np.std(a) == 0:
        return 0.0
    return float(np.corrcoef(p, a)[0, 1])


def get_rolling_windows(dates, train_months=48, val_months=6):
    dates = pd.DatetimeIndex(dates).sort_values()
    months = dates.to_period('M').unique().sort_values()

    windows = []
    total_needed = train_months + val_months + 1
    for i in range(total_needed, len(months) + 1):
        test_month = months[i - 1]
        val_months_range = months[i - 1 - val_months:i - 1]
        train_months_range = months[i - total_needed:i - 1 - val_months]

        train_dates = dates[dates.to_period('M').isin(train_months_range)]
        val_dates = dates[dates.to_period('M').isin(val_months_range)]
        test_dates = dates[dates.to_period('M') == test_month]

        if len(train_dates) == 0 or len(val_dates) == 0 or len(test_dates) == 0:
            continue

        windows.append({
            'train_start': train_dates.min(),
            'train_end': train_dates.max(),
            'val_start': val_dates.min(),
            'val_end': val_dates.max(),
            'test_start': test_dates.min(),
            'test_end': test_dates.max(),
            'model_date': test_dates.max(),
        })
    return windows


def fit_linear_ridge(X_train, y_train, X_val, y_val):
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xv = scaler.transform(X_val)
    best = (-np.inf, None, None)
    for a in np.logspace(-4, 1.5, 30):
        m = Ridge(alpha=float(a), fit_intercept=True, max_iter=10000)
        m.fit(Xt, y_train)
        ic = compute_ic(m.predict(Xv), y_val)
        if ic > best[0]:
            best = (ic, float(a), m)

    Xfull = np.vstack([X_train, X_val])
    yfull = np.concatenate([y_train, y_val])
    scaler = StandardScaler().fit(Xfull)
    Xf = scaler.transform(Xfull)
    m = Ridge(alpha=best[1], fit_intercept=True, max_iter=10000)
    m.fit(Xf, yfull)

    coef = m.coef_.astype(np.float32)
    scale = np.where(scaler.scale_.astype(np.float32) == 0, 1.0, scaler.scale_.astype(np.float32))
    mean = scaler.mean_.astype(np.float32)
    w = coef / scale
    c = -float(np.dot(mean / scale, coef))
    return {'val_ic': best[0], 'bundle': LinearModelBundle(None, w, c), 'alpha': best[1], 'kind': 'linear'}


def fit_linear_elasticnet(X_train, y_train, X_val, y_val):
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    alphas = np.logspace(-5, 0, 16)
    l1s = [0.1, 0.3, 0.5, 0.7, 0.9]
    best = (-np.inf, None, None, None)
    for a in alphas:
        for l1 in l1s:
            m = ElasticNet(alpha=float(a), l1_ratio=float(l1), fit_intercept=True, max_iter=20000)
            m.fit(Xt, y_train)
            ic = compute_ic(m.predict(Xv), y_val)
            if ic > best[0]:
                best = (ic, float(a), float(l1), m)

    Xfull = np.vstack([X_train, X_val])
    yfull = np.concatenate([y_train, y_val])
    scaler = StandardScaler().fit(Xfull)
    Xf = scaler.transform(Xfull)
    m = ElasticNet(alpha=best[1], l1_ratio=best[2], fit_intercept=True, max_iter=20000)
    m.fit(Xf, yfull)

    coef = m.coef_.astype(np.float32)
    scale = np.where(scaler.scale_.astype(np.float32) == 0, 1.0, scaler.scale_.astype(np.float32))
    mean = scaler.mean_.astype(np.float32)
    w = coef / scale
    c = -float(np.dot(mean / scale, coef))
    return {'val_ic': best[0], 'bundle': LinearModelBundle(None, w, c), 'alpha': best[1], 'l1_ratio': best[2], 'kind': 'linear'}


def fit_linear_pcr(X_train, y_train, X_val, y_val, pcr_max_components=None):
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    max_comp = min(Xt.shape[0], Xt.shape[1])
    if pcr_max_components is not None:
        max_comp = min(max_comp, int(pcr_max_components))
    max_comp = max(1, int(max_comp))
    grid = sorted(set([1, 2, 5, 10, 20, 40, 80, 120, 200, max_comp]))
    grid = [g for g in grid if 1 <= g <= max_comp]

    # Fit one decomposition, then sweep k using orthogonal-PC OLS in PC space.
    pca_all = PCA(n_components=max_comp, svd_solver='randomized', random_state=42)
    Zt_all = pca_all.fit_transform(Xt)
    Zv_all = pca_all.transform(Xv)
    zt_norm2 = np.sum(Zt_all * Zt_all, axis=0)
    zt_norm2 = np.where(zt_norm2 == 0.0, 1.0, zt_norm2)
    y_train_arr = np.asarray(y_train, dtype=np.float64)

    best = (-np.inf, None)
    for k in grid:
        beta = (Zt_all[:, :k].T @ y_train_arr) / zt_norm2[:k]
        pred = Zv_all[:, :k] @ beta
        ic = compute_ic(pred, y_val)
        if ic > best[0]:
            best = (ic, int(k))

    Xfull = np.vstack([X_train, X_val])
    yfull = np.concatenate([y_train, y_val])
    scaler = StandardScaler().fit(Xfull)
    Xf = scaler.transform(Xfull)
    max_comp_full = max(1, min(Xf.shape[0], Xf.shape[1]))
    if pcr_max_components is not None:
        max_comp_full = min(max_comp_full, int(pcr_max_components))
    n_comp = min(best[1], max_comp_full)
    pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=42)
    Zf = pca.fit_transform(Xf)
    zf_norm2 = np.sum(Zf * Zf, axis=0)
    zf_norm2 = np.where(zf_norm2 == 0.0, 1.0, zf_norm2)
    beta = (Zf.T @ np.asarray(yfull, dtype=np.float64)) / zf_norm2

    w_scaled = pca.components_.T @ beta.astype(np.float32)
    scale = np.where(scaler.scale_.astype(np.float32) == 0, 1.0, scaler.scale_.astype(np.float32))
    mean = scaler.mean_.astype(np.float32)
    w = w_scaled / scale
    c = -float(np.dot(mean / scale, w_scaled))
    return {'val_ic': best[0], 'bundle': LinearModelBundle(None, w, c), 'n_components': int(n_comp), 'kind': 'linear'}


def _winsorize_with_train_bounds(X_train, X_other, lower_q=0.005, upper_q=0.995):
    lo = np.nanquantile(X_train, lower_q, axis=0)
    hi = np.nanquantile(X_train, upper_q, axis=0)
    return np.clip(X_train, lo, hi), np.clip(X_other, lo, hi)


def fit_linear_robust_pcr(X_train, y_train, X_val, y_val):
    X_train_w, X_val_w = _winsorize_with_train_bounds(X_train, X_val, lower_q=0.005, upper_q=0.995)
    scaler = StandardScaler().fit(X_train_w)
    Xt = scaler.transform(X_train_w)
    Xv = scaler.transform(X_val_w)

    max_comp = max(1, min(Xt.shape[0], Xt.shape[1]))
    # Keep tuned search, but use a compact grid to keep walk-forward runtime tractable.
    comp_grid = sorted(set([1, 2, 5, 10, 20, 40, 80, max_comp]))
    comp_grid = [k for k in comp_grid if 1 <= k <= max_comp]
    alpha_grid = [1e-5, 1e-4, 1e-3, 1e-2]

    best = (-np.inf, None, None)
    for k in comp_grid:
        pca = PCA(n_components=k, svd_solver='randomized', random_state=42)
        Zt = pca.fit_transform(Xt)
        Zv = pca.transform(Xv)
        for alpha in alpha_grid:
            reg = HuberRegressor(epsilon=1.35, alpha=float(alpha), fit_intercept=True, max_iter=300)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                reg.fit(Zt, y_train)
            ic = compute_ic(reg.predict(Zv), y_val)
            if ic > best[0]:
                best = (ic, k, float(alpha))

    Xfull = np.vstack([X_train, X_val])
    yfull = np.concatenate([y_train, y_val])
    Xfull_w, _ = _winsorize_with_train_bounds(Xfull, Xfull, lower_q=0.005, upper_q=0.995)
    scaler = StandardScaler().fit(Xfull_w)
    Xf = scaler.transform(Xfull_w)
    n_comp = min(best[1], max(1, min(Xf.shape[0], Xf.shape[1])))
    pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=42)
    Zf = pca.fit_transform(Xf)
    reg = HuberRegressor(epsilon=1.35, alpha=best[2], fit_intercept=True, max_iter=300)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(Zf, yfull)

    w_scaled = pca.components_.T @ reg.coef_.astype(np.float32)
    scale = np.where(scaler.scale_.astype(np.float32) == 0, 1.0, scaler.scale_.astype(np.float32))
    mean = scaler.mean_.astype(np.float32)
    w = w_scaled / scale
    c = -float(np.dot(mean / scale, w_scaled))
    return {
        'val_ic': best[0],
        'bundle': LinearModelBundle(None, w, c),
        'n_components': int(n_comp),
        'alpha': float(best[2]),
        'kind': 'linear',
    }


def fit_linear_huber(X_train, y_train, X_val, y_val):
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    eps_grid = [1.2, 1.35, 1.5, 1.75, 2.0]
    alpha_grid = [1e-5, 1e-4, 1e-3, 1e-2]
    best = (-np.inf, None, None)
    for eps in eps_grid:
        for alpha in alpha_grid:
            m = HuberRegressor(epsilon=eps, alpha=alpha, fit_intercept=True, max_iter=500)
            m.fit(Xt, y_train)
            ic = compute_ic(m.predict(Xv), y_val)
            if ic > best[0]:
                best = (ic, (eps, alpha), m)

    Xfull = np.vstack([X_train, X_val])
    yfull = np.concatenate([y_train, y_val])
    scaler = StandardScaler().fit(Xfull)
    Xf = scaler.transform(Xfull)
    eps, alpha = best[1]
    m = HuberRegressor(epsilon=eps, alpha=alpha, fit_intercept=True, max_iter=500)
    m.fit(Xf, yfull)

    coef = m.coef_.astype(np.float32)
    scale = np.where(scaler.scale_.astype(np.float32) == 0, 1.0, scaler.scale_.astype(np.float32))
    mean = scaler.mean_.astype(np.float32)
    w = coef / scale
    c = -float(np.dot(mean / scale, coef))
    return {'val_ic': best[0], 'bundle': LinearModelBundle(None, w, c), 'epsilon': eps, 'alpha': alpha, 'kind': 'linear'}


def fit_linear_pls(X_train, y_train, X_val, y_val):
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    max_comp = min(Xt.shape[0] - 1, Xt.shape[1])
    max_comp = max(1, int(max_comp))
    grid = sorted(set([1, 2, 3, 5, 10, 20, 40, 80, max_comp]))
    grid = [g for g in grid if 1 <= g <= max_comp]

    best = (-np.inf, None, None)
    for k in grid:
        m = PLSRegression(n_components=int(k), scale=False, max_iter=500)
        m.fit(Xt, y_train)
        pred = np.asarray(m.predict(Xv)).reshape(-1)
        ic = compute_ic(pred, y_val)
        if ic > best[0]:
            best = (ic, int(k), m)

    Xfull = np.vstack([X_train, X_val])
    yfull = np.concatenate([y_train, y_val])
    scaler = StandardScaler().fit(Xfull)
    Xf = scaler.transform(Xfull)
    k = min(best[1], max(1, min(Xf.shape[0] - 1, Xf.shape[1])))
    m = PLSRegression(n_components=int(k), scale=False, max_iter=500)
    m.fit(Xf, yfull)

    coef_arr = np.asarray(m.coef_)
    if coef_arr.ndim == 2:
        if coef_arr.shape[0] == Xf.shape[1]:
            coef = coef_arr[:, 0]
        elif coef_arr.shape[1] == Xf.shape[1]:
            coef = coef_arr[0, :]
        else:
            coef = coef_arr.reshape(-1)[:Xf.shape[1]]
    else:
        coef = coef_arr.reshape(-1)[:Xf.shape[1]]
    coef = coef.astype(np.float32)

    scale = np.where(scaler.scale_.astype(np.float32) == 0, 1.0, scaler.scale_.astype(np.float32))
    mean = scaler.mean_.astype(np.float32)
    y_mean = float(np.asarray(getattr(m, '_y_mean', np.mean(yfull))).reshape(-1)[0])
    w = coef / scale
    c = y_mean - float(np.dot(mean / scale, coef))
    return {'val_ic': best[0], 'bundle': LinearModelBundle(None, w, c), 'n_components': int(k), 'kind': 'linear'}


def fit_rf(X_train, y_train, X_val, y_val):
    params_grid = [
        {'n_estimators': 200, 'max_depth': 6, 'min_samples_leaf': 20},
        {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 20},
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 10},
    ]
    best = (-np.inf, None, None)
    for p in params_grid:
        m = RandomForestRegressor(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            min_samples_leaf=p['min_samples_leaf'],
            n_jobs=-1,
            random_state=42,
        )
        m.fit(X_train, y_train)
        ic = compute_ic(m.predict(X_val), y_val)
        if ic > best[0]:
            best = (ic, p, m)

    Xfull = np.vstack([X_train, X_val])
    yfull = np.concatenate([y_train, y_val])
    m = RandomForestRegressor(
        n_estimators=best[1]['n_estimators'],
        max_depth=best[1]['max_depth'],
        min_samples_leaf=best[1]['min_samples_leaf'],
        n_jobs=-1,
        random_state=42,
    )
    m.fit(Xfull, yfull)
    return {'val_ic': best[0], 'model': m, 'rf_params': best[1], 'kind': 'rf'}


def fit_rrr_group(X_train, Y_train, X_val, Y_val):
    scaler = StandardScaler().fit(X_train)
    Xt = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    # OLS in scaled space
    B_ols, *_ = np.linalg.lstsq(Xt, Y_train, rcond=None)
    Yhat_t = Xt @ B_ols

    U, S, Vt = np.linalg.svd(Yhat_t, full_matrices=False)
    q = Y_train.shape[1]
    ranks = sorted(set([1, 2, 3, 5, min(8, q), min(12, q), q]))
    ranks = [r for r in ranks if 1 <= r <= q]

    best = (-np.inf, None, None)
    for r in ranks:
        Vr = Vt[:r, :].T
        B_r = B_ols @ (Vr @ Vr.T)
        pred = Xv @ B_r
        ic_list = []
        for j in range(Y_val.shape[1]):
            ic_list.append(compute_ic(pred[:, j], Y_val[:, j]))
        mean_ic = float(np.nanmean(ic_list))
        if mean_ic > best[0]:
            best = (mean_ic, r, B_r)

    # refit on full and keep rank
    Xfull = np.vstack([X_train, X_val])
    Yfull = np.vstack([Y_train, Y_val])
    scaler = StandardScaler().fit(Xfull)
    Xf = scaler.transform(Xfull)
    B_ols, *_ = np.linalg.lstsq(Xf, Yfull, rcond=None)
    Yhat = Xf @ B_ols
    U, S, Vt = np.linalg.svd(Yhat, full_matrices=False)
    Vr = Vt[:best[1], :].T
    B_r = B_ols @ (Vr @ Vr.T)

    scale = np.where(scaler.scale_.astype(np.float32) == 0, 1.0, scaler.scale_.astype(np.float32))
    mean = scaler.mean_.astype(np.float32)

    # convert each output column to raw-space linear params
    bundles = []
    for j in range(B_r.shape[1]):
        w_scaled = B_r[:, j].astype(np.float32)
        w = w_scaled / scale
        c = -float(np.dot(mean / scale, w_scaled))
        bundles.append((w, c))

    return {'val_ic': best[0], 'rank': int(best[1]), 'bundles': bundles, 'kind': 'linear'}


def train_single_ticker(
    model_kind,
    ticker,
    features_df,
    output_dir,
    train_months,
    val_months,
    snapshot_dates,
    constituent_sets,
    turnover_df,
    min_turnover,
    min_model_date=None,
    pcr_max_components=None,
    skip_existing=False,
):
    y = features_df['ordinary_residual']
    all_feature_cols = [c for c in features_df.columns if c.startswith('russell_')]
    if not all_feature_cols:
        return []

    windows = get_rolling_windows(features_df.index, train_months=train_months, val_months=val_months)
    if not windows:
        return []

    tdir = output_dir / ticker
    tdir.mkdir(parents=True, exist_ok=True)
    md = []

    for w in windows:
        if min_model_date is not None and pd.Timestamp(w['model_date']) < min_model_date:
            continue
        out = tdir / f"{w['model_date'].strftime('%Y_%m')}.pkl"
        if skip_existing and out.exists():
            continue

        tv_mask = (features_df.index >= w['train_start']) & (features_df.index <= w['val_end'])
        tv_dates = features_df.index[tv_mask]
        feature_cols = select_window_feature_columns(
            all_feature_cols,
            tv_dates,
            w['train_start'],
            w['val_end'],
            snapshot_dates,
            constituent_sets,
            turnover_df,
            min_turnover,
        )
        if len(feature_cols) < 5:
            continue
        X = features_df[feature_cols]

        train_mask = (features_df.index >= w['train_start']) & (features_df.index <= w['train_end'])
        val_mask = (features_df.index >= w['val_start']) & (features_df.index <= w['val_end'])
        test_mask = (features_df.index >= w['test_start']) & (features_df.index <= w['test_end'])

        X_train = X[train_mask].values
        y_train = y[train_mask].values
        X_val = X[val_mask].values
        y_val = y[val_mask].values
        X_test = X[test_mask].values
        y_test = y[test_mask].values

        if len(y_train) < 120 or len(y_val) < 30 or len(y_test) < 10:
            continue

        if model_kind == 'ridge':
            fit = fit_linear_ridge(X_train, y_train, X_val, y_val)
            y_pred = fit['bundle'].predict_raw(X_test)
        elif model_kind == 'pcr':
            fit = fit_linear_pcr(
                X_train,
                y_train,
                X_val,
                y_val,
                pcr_max_components=pcr_max_components,
            )
            y_pred = fit['bundle'].predict_raw(X_test)
        elif model_kind == 'robust_pcr':
            fit = fit_linear_robust_pcr(X_train, y_train, X_val, y_val)
            y_pred = fit['bundle'].predict_raw(X_test)
        elif model_kind == 'elasticnet':
            fit = fit_linear_elasticnet(X_train, y_train, X_val, y_val)
            y_pred = fit['bundle'].predict_raw(X_test)
        elif model_kind == 'pls':
            fit = fit_linear_pls(X_train, y_train, X_val, y_val)
            y_pred = fit['bundle'].predict_raw(X_test)
        elif model_kind == 'rf':
            fit = fit_rf(X_train, y_train, X_val, y_val)
            y_pred = fit['model'].predict(X_test)
        elif model_kind == 'huber':
            fit = fit_linear_huber(X_train, y_train, X_val, y_val)
            y_pred = fit['bundle'].predict_raw(X_test)
        else:
            raise ValueError(f'Unsupported single-ticker model kind: {model_kind}')

        # Persist comparable model/baseline diagnostics for downstream signal gating.
        # Residual baseline prediction is 0, so baseline IC evaluates to 0.0 by definition here.
        X_fit = np.vstack([X_train, X_val])
        y_fit = np.concatenate([y_train, y_val])
        if fit['kind'] == 'linear':
            y_fit_pred = fit['bundle'].predict_raw(X_fit)
        else:
            y_fit_pred = fit['model'].predict(X_fit)
        train_ic = compute_ic(y_fit_pred, y_fit)
        baseline_train_ic = 0.0
        baseline_val_ic = 0.0

        test_ic = compute_ic(y_pred, y_test)
        window_constituent_start = _last_snapshot_on_or_before(snapshot_dates, w['train_start'])
        window_constituent_end = _last_snapshot_on_or_before(snapshot_dates, w['val_end'])

        model_data = {
            'kind': fit['kind'],
            'feature_names': feature_cols,
            'feature_tickers': [c.replace('russell_', '', 1) for c in feature_cols],
            'feature_index_map': {c.replace('russell_', '', 1): i for i, c in enumerate(feature_cols)},
            'window_constituent_start': window_constituent_start,
            'window_constituent_end': window_constituent_end,
            'min_turnover': float(min_turnover),
            'train_period': (w['train_start'], w['train_end']),
            'val_period': (w['val_start'], w['val_end']),
            'test_period': (w['test_start'], w['test_end']),
            'model_date': w['model_date'],
            'train_ic': train_ic,
            'baseline_train_ic': baseline_train_ic,
            'val_ic': fit['val_ic'],
            'baseline_val_ic': baseline_val_ic,
            'test_ic': test_ic,
            'n_features': len(feature_cols),
        }
        if fit['kind'] == 'linear':
            fit['bundle'].feature_names = feature_cols
            model_data['w_raw'] = fit['bundle'].w_raw
            model_data['c_raw'] = fit['bundle'].c_raw
        else:
            model_data['model'] = fit['model']

        for k in ['alpha', 'l1_ratio', 'n_components', 'rf_params', 'epsilon']:
            if k in fit:
                model_data[k] = fit[k]

        with open(out, 'wb') as f:
            pickle.dump(model_data, f)

        md.append({
            'ticker': ticker,
            'model_date': w['model_date'],
            'train_ic': train_ic,
            'baseline_train_ic': baseline_train_ic,
            'val_ic': fit['val_ic'],
            'baseline_val_ic': baseline_val_ic,
            'test_ic': test_ic,
            'model_file': str(out),
        })

    return md


def train_rrr(features_dir, output_dir, train_months, val_months, universe, min_model_date=None):
    # load all features once
    feats = {}
    for fp in sorted(features_dir.glob('*.parquet')):
        t = fp.stem
        if t not in universe:
            continue
        feats[t] = pd.read_parquet(fp)

    if not feats:
        return []

    # group by market index via adr_info/futures mapping
    adr_info = pd.read_csv(Path('data/raw/adr_info.csv'))
    adr_info['adr'] = adr_info['adr'].str.replace(' US Equity', '', regex=False)
    group_map = adr_info.set_index('adr')['index_future_bbg'].to_dict()

    idx_groups = {}
    for t in feats:
        g = group_map.get(t)
        if g is None:
            continue
        idx_groups.setdefault(g, []).append(t)

    all_md = []
    for g, tickers in idx_groups.items():
        tickers = sorted([t for t in tickers if t in feats])
        if len(tickers) < 2:
            continue

        # align on common dates and common features
        common_dates = feats[tickers[0]].index
        for t in tickers[1:]:
            common_dates = common_dates.intersection(feats[t].index)
        if len(common_dates) < 500:
            continue

        feature_cols = sorted(list(set.intersection(*[set([c for c in feats[t].columns if c.startswith('russell_')]) for t in tickers])))
        if len(feature_cols) < 20:
            continue

        X_df = feats[tickers[0]].loc[common_dates, feature_cols]
        Y_df = pd.concat([feats[t].loc[common_dates, 'ordinary_residual'].rename(t) for t in tickers], axis=1)

        windows = get_rolling_windows(common_dates, train_months=train_months, val_months=val_months)
        if not windows:
            continue

        for w in windows:
            if min_model_date is not None and pd.Timestamp(w['model_date']) < min_model_date:
                continue

            train_mask = (X_df.index >= w['train_start']) & (X_df.index <= w['train_end'])
            val_mask = (X_df.index >= w['val_start']) & (X_df.index <= w['val_end'])
            test_mask = (X_df.index >= w['test_start']) & (X_df.index <= w['test_end'])

            X_train = X_df[train_mask].values
            Y_train = Y_df[train_mask].values
            X_val = X_df[val_mask].values
            Y_val = Y_df[val_mask].values
            X_test = X_df[test_mask].values
            Y_test = Y_df[test_mask].values

            if X_train.shape[0] < 120 or X_val.shape[0] < 30 or X_test.shape[0] < 10:
                continue

            fit = fit_rrr_group(X_train, Y_train, X_val, Y_val)

            # save per ticker model bundle for easy inference reuse
            for j, t in enumerate(tickers):
                tdir = output_dir / t
                tdir.mkdir(parents=True, exist_ok=True)
                w_raw, c_raw = fit['bundles'][j]
                y_pred = X_test @ w_raw + c_raw
                test_ic = compute_ic(y_pred, Y_test[:, j])

                model_data = {
                    'kind': 'linear',
                    'feature_names': feature_cols,
                    'w_raw': np.asarray(w_raw, dtype=np.float32),
                    'c_raw': float(c_raw),
                    'rank': fit['rank'],
                    'group': g,
                    'group_tickers': tickers,
                    'train_period': (w['train_start'], w['train_end']),
                    'val_period': (w['val_start'], w['val_end']),
                    'test_period': (w['test_start'], w['test_end']),
                    'model_date': w['model_date'],
                    'val_ic': fit['val_ic'],
                    'test_ic': test_ic,
                }

                out = tdir / f"{w['model_date'].strftime('%Y_%m')}.pkl"
                with open(out, 'wb') as f:
                    pickle.dump(model_data, f)

                all_md.append({
                    'ticker': t,
                    'model_date': w['model_date'],
                    'val_ic': fit['val_ic'],
                    'test_ic': test_ic,
                    'rank': fit['rank'],
                    'group': g,
                    'model_file': str(out),
                })

    return all_md


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-kind', choices=['ridge', 'pcr', 'robust_pcr', 'elasticnet', 'pls', 'rf', 'rrr', 'huber'], required=True)
    p.add_argument('--features-dir', default='data/processed/models/with_us_stocks/features_extended')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--train-months', type=int, default=48)
    p.add_argument('--val-months', type=int, default=6)
    p.add_argument('--min-model-date', type=str, default=None)
    p.add_argument('--historical-constituents-file', default='data/raw/historical_russell_1000.csv')
    p.add_argument('--turnover-file', default='data/raw/russell1000/russell1000_turnover.csv')
    p.add_argument('--min-turnover', type=float, default=10_000_000.0)
    p.add_argument('--pcr-max-components', type=int, default=None)
    p.add_argument('--skip-existing', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_model_date = pd.Timestamp(args.min_model_date) if args.min_model_date else None
    snapshot_dates, constituent_sets = load_historical_russell_constituents(Path(args.historical_constituents_file))
    turnover_df = load_turnover(Path(args.turnover_file))

    all_md = []
    experiment_universe = load_experiment_universe()

    if args.model_kind == 'rrr':
        all_md = train_rrr(
            features_dir, output_dir, args.train_months, args.val_months, experiment_universe, min_model_date=min_model_date
        )
    else:
        for fp in sorted(features_dir.glob('*.parquet')):
            ticker = fp.stem
            if ticker not in experiment_universe:
                continue
            try:
                features = pd.read_parquet(fp)
                md = train_single_ticker(
                    model_kind=args.model_kind,
                    ticker=ticker,
                    features_df=features,
                    output_dir=output_dir,
                    train_months=args.train_months,
                    val_months=args.val_months,
                    snapshot_dates=snapshot_dates,
                    constituent_sets=constituent_sets,
                    turnover_df=turnover_df,
                    min_turnover=args.min_turnover,
                    min_model_date=min_model_date,
                    pcr_max_components=args.pcr_max_components,
                    skip_existing=args.skip_existing,
                )
                all_md.extend(md)
            except Exception as e:
                print(f'Error training {ticker}: {e}')

    if all_md:
        mdf = pd.DataFrame(all_md)
        mdf.to_csv(output_dir / 'training_metadata.csv', index=False)
        print(f"Trained {len(mdf)} models for {mdf['ticker'].nunique()} tickers -> {output_dir}")
    else:
        print('No models trained')


if __name__ == '__main__':
    main()
