# ml.py
import os
import time
import uuid

import joblib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

from config import (
    ML_HORIZON_SECONDS, ML_FLAT_THRESHOLD, ML_MIN_TRAIN_SAMPLES,
    ML_N_SPLITS, ML_MODELS_DIR, FEATURE_SCHEMA_VERSION,
    ML_CALIBRATION,
)


FEATURE_COLS = [
    "return_1", "return_5", "return_20",
    "rsi", "ema20", "ema50",
    "volatility", "volume_ratio",
    "high_low_range", "body_size", "upper_wick", "lower_wick",
]

LABEL_DOWN = 0
LABEL_FLAT = 1
LABEL_UP = 2

DIR_NAMES = {LABEL_DOWN: "DOWN", LABEL_FLAT: "FLAT", LABEL_UP: "UP"}


def _timeframe_to_ms(timeframe):
    unit = timeframe[-1]
    n = int(timeframe[:-1])
    mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]
    return n * mult


def horizon_index(timeframe, horizon_seconds):
    tf_ms = _timeframe_to_ms(timeframe)
    return max(1, int(round(horizon_seconds * 1000 / tf_ms)))


def build_target(candles_arr, horizon_idx, flat_threshold):
    close = candles_arr[:, 4]
    n = close.size
    if n <= horizon_idx:
        return None, None
    future = close[horizon_idx:]
    now = close[:-horizon_idx]
    y_ret = future / now - 1.0
    y_dir = np.full(y_ret.size, LABEL_FLAT, dtype=np.int8)
    y_dir[y_ret > flat_threshold] = LABEL_UP
    y_dir[y_ret < -flat_threshold] = LABEL_DOWN
    return y_dir, y_ret


def _make_classifier():
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=-1,
    )


def _make_regressor():
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=-1,
    )


def _walk_forward_validate(X, y_dir, y_ret, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, maes = [], []
    for tr, va in tscv.split(X):
        if len(tr) < 50:
            continue
        cls = _make_classifier()
        cls.fit(X[tr], y_dir[tr])
        pred = cls.predict(X[va])
        accs.append(float((pred == y_dir[va]).mean()))

        reg = _make_regressor()
        reg.fit(X[tr], y_ret[tr])
        p = reg.predict(X[va])
        maes.append(float(np.abs(p - y_ret[va]).mean()))
    return accs, maes


def _prepare_xy(candles_arr, X, flat_threshold, horizon_seconds, timeframe):
    horizon_idx = horizon_index(timeframe, horizon_seconds)
    y_dir, y_ret = build_target(candles_arr, horizon_idx, flat_threshold)
    if y_dir is None:
        raise ValueError("Not enough candles for the given horizon")
    X = X[:y_dir.size]
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_ret)
    X, y_dir, y_ret = X[mask], y_dir[mask], y_ret[mask]
    return X, y_dir, y_ret, horizon_idx


def _fit_calibrated(X, y_dir):
    base = _make_classifier()
    if not ML_CALIBRATION:
        base.fit(X, y_dir)
        return base

    tscv = TimeSeriesSplit(n_splits=3)
    try:
        cal = CalibratedClassifierCV(
            base, method=ML_CALIBRATION, cv=tscv)
        cal.fit(X, y_dir)
        return cal
    except Exception:
        base.fit(X, y_dir)
        return base


def feature_importance(cls_model):
    try:
        if isinstance(cls_model, CalibratedClassifierCV):
            boosters = [est.estimator for est in cls_model.calibrated_classifiers_]
            imps = np.zeros(len(FEATURE_COLS))
            for b in boosters:
                imps += b.feature_importances_
            imps /= max(1, len(boosters))
        else:
            imps = cls_model.feature_importances_
        imps = np.asarray(imps, dtype=float)
        total = imps.sum()
        if total > 0:
            imps = imps / total
        return [{"feature": FEATURE_COLS[i], "importance": float(imps[i])}
                for i in range(len(FEATURE_COLS))]
    except Exception:
        return []


def train_model(symbol, timeframe, candles_arr, X,
                flat_threshold=ML_FLAT_THRESHOLD,
                horizon_seconds=ML_HORIZON_SECONDS):
    X, y_dir, y_ret, horizon_idx = _prepare_xy(
        candles_arr, X, flat_threshold, horizon_seconds, timeframe)

    if X.shape[0] < ML_MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Not enough clean samples: {X.shape[0]} < {ML_MIN_TRAIN_SAMPLES}")

    accs, maes = _walk_forward_validate(X, y_dir, y_ret, ML_N_SPLITS)
    cls_accuracy = float(np.mean(accs)) if accs else None
    reg_mae = float(np.mean(maes)) if maes else None

    cls = _fit_calibrated(X, y_dir)
    reg = _make_regressor()
    reg.fit(X, y_ret)

    os.makedirs(ML_MODELS_DIR, exist_ok=True)
    version = (
        f"{symbol}_{timeframe}_h{horizon_seconds}_"
        f"{FEATURE_SCHEMA_VERSION}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    cls_path = os.path.join(ML_MODELS_DIR, f"{version}_cls.pkl")
    reg_path = os.path.join(ML_MODELS_DIR, f"{version}_reg.txt")
    joblib.dump(cls, cls_path)
    reg.booster_.save_model(reg_path)

    return {
        "version": version,
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_seconds": horizon_seconds,
        "flat_threshold": flat_threshold,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "n_train": int(X.shape[0]),
        "cls_accuracy": cls_accuracy,
        "reg_mae": reg_mae,
        "trained_at": int(time.time()),
        "cls_path": cls_path,
        "reg_path": reg_path,
    }


def load_boosters(cls_path, reg_path):
    if cls_path.endswith(".pkl"):
        cls = joblib.load(cls_path)
    else:
        cls = lgb.Booster(model_file=cls_path)
    reg = lgb.Booster(model_file=reg_path)
    return cls, reg


def _predict_proba(cls, x):
    if isinstance(cls, lgb.Booster):
        p = cls.predict(x)
        if p.ndim == 1:
            return np.array([[1 - p, p]])
        return p
    return cls.predict_proba(x)


def predict_latest(candles_arr, X, cls, reg):
    if X is None or X.size == 0:
        return None
    mask = ~np.isnan(X).any(axis=1)
    if not mask.any():
        return None
    idx = int(np.where(mask)[0][-1])
    x = X[idx:idx + 1]

    proba = _predict_proba(cls, x)[0]
    pred_ret = float(reg.predict(x)[0])

    label = int(np.argmax(proba))
    return {
        "idx": idx,
        "direction": DIR_NAMES[label],
        "probability": float(proba[label]),
        "proba": [float(p) for p in proba],
        "expected_return": pred_ret,
        "close": float(candles_arr[idx, 4]),
        "ts_ms": int(candles_arr[idx, 0]),
    }


def shap_explain(cls, X, feature_cols, background_size, max_rows,
                 top_k=8):
    try:
        import shap
    except Exception:
        return None

    mask = ~np.isnan(X).any(axis=1)
    Xv = X[mask]
    if Xv.shape[0] < 50:
        return None
    if Xv.shape[0] > max_rows:
        Xv = Xv[-max_rows:]

    bg = Xv[-min(background_size, Xv.shape[0]):]
    x_target = Xv[-1:]

    try:
        raw = cls
        if isinstance(cls, CalibratedClassifierCV):
            raw = cls.calibrated_classifiers_[0].estimator

        explainer = shap.TreeExplainer(raw)
        shap_values = explainer.shap_values(x_target)
    except Exception:
        return None

    if isinstance(shap_values, list):
        contribs = shap_values
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            contribs = [arr[0, :, i] for i in range(arr.shape[2])]
        else:
            contribs = [arr[0]]

    per_class = []
    for k, c in enumerate(contribs):
        c = np.asarray(c).ravel()
        pairs = list(zip(feature_cols, c.tolist()))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        per_class.append({
            "label": DIR_NAMES.get(k, str(k)),
            "top": [{"feature": f, "value": float(v)}
                    for f, v in pairs[:top_k]],
            "base_value": None,
        })
    return per_class