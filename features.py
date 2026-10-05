# features.py
import numpy as np

from config import EMA_FAST, EMA_SLOW, RSI_PERIOD, VOL_WINDOW, VOLUME_WINDOW


def ema(values, period):
    n = values.size
    out = np.full(n, np.nan)
    if n < period:
        return out
    k = 2.0 / (period + 1.0)
    seed = values[:period].mean()
    out[period - 1] = seed
    for i in range(period, n):
        out[i] = values[i] * k + out[i - 1] * (1.0 - k)
    return out


def rsi(closes, period=RSI_PERIOD):
    n = closes.size
    out = np.full(n, np.nan)
    if n <= period:
        return out

    diff = np.diff(closes)
    gains = np.where(diff >= 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    def _rsi(ag, al):
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - 100.0 / (1.0 + rs)

    out[period] = _rsi(avg_gain, avg_loss)
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = _rsi(avg_gain, avg_loss)
    return out


def rolling_return(closes, lag):
    n = closes.size
    out = np.full(n, np.nan)
    if n <= lag:
        return out
    prev = closes[:-lag]
    curr = closes[lag:]
    out[lag:] = np.where(prev > 0, curr / prev - 1.0, np.nan)
    return out


def rolling_std_returns(closes, window):
    n = closes.size
    out = np.full(n, np.nan)
    if n < window + 1:
        return out
    ret = np.empty(n)
    ret[0] = np.nan
    ret[1:] = closes[1:] / closes[:-1] - 1.0
    for i in range(window, n):
        out[i] = ret[i - window + 1:i + 1].std(ddof=0)
    return out


def rolling_mean(values, window):
    n = values.size
    out = np.full(n, np.nan)
    if n < window:
        return out
    csum = np.cumsum(values, dtype=float)
    out[window - 1] = csum[window - 1] / window
    out[window:] = (csum[window:] - csum[:-window]) / window
    return out


def compute_features(candles):
    o = candles[:, 1]
    h = candles[:, 2]
    l = candles[:, 3]
    c = candles[:, 4]
    v = candles[:, 5]

    ema20 = ema(c, EMA_FAST)
    ema50 = ema(c, EMA_SLOW)
    rsi14 = rsi(c, RSI_PERIOD)

    return_1 = rolling_return(c, 1)
    return_5 = rolling_return(c, 5)
    return_20 = rolling_return(c, 20)

    volatility = rolling_std_returns(c, VOL_WINDOW)

    vmean = rolling_mean(v, VOLUME_WINDOW)
    volume_ratio = np.where(vmean > 0, v / vmean, np.nan)

    safe_close = np.where(c > 0, c, np.nan)
    high_low_range = (h - l) / safe_close
    body_size = np.abs(c - o) / safe_close
    upper_wick = (h - np.maximum(o, c)) / safe_close
    lower_wick = (np.minimum(o, c) - l) / safe_close

    return {
        "return_1": return_1,
        "return_5": return_5,
        "return_20": return_20,
        "rsi": rsi14,
        "ema20": ema20,
        "ema50": ema50,
        "volatility": volatility,
        "volume_ratio": volume_ratio,
        "high_low_range": high_low_range,
        "body_size": body_size,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
    }


def compute_indicators(candles):
    arr = np.asarray(candles, dtype=float)
    if arr.size == 0:
        return {"ts": [], "close": [],
                "ema_fast": [], "ema_slow": [], "rsi": []}

    f = compute_features(arr)

    def to_list(a):
        return [None if np.isnan(x) else float(x) for x in a]

    return {
        "ts": arr[:, 0].astype(np.int64).tolist(),
        "close": arr[:, 4].tolist(),
        "ema_fast": to_list(f["ema20"]),
        "ema_slow": to_list(f["ema50"]),
        "rsi": to_list(f["rsi"]),
    }