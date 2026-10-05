# simulation.py
import json
import time

from config import (
    SIMULATION_FEE_PCT,
    SIMULATION_STOP_LOSS_PCT,
    SIMULATION_TAKE_PROFIT_PCT,
    SIMULATION_SIZING_MODE,
    SIMULATION_POSITION_PCT,
    SIMULATION_KELLY_FRACTION,
    SIMULATION_FIXED_SIZE_USDT,
)


def _tf_ms(timeframe):
    unit = timeframe[-1]
    n = int(timeframe[:-1])
    return {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit] * n


def _position_size(mode, capital, price, win_rate, avg_win, avg_loss):
    if mode == "fixed":
        return min(SIMULATION_FIXED_SIZE_USDT, capital)
    if mode == "percent":
        return capital * SIMULATION_POSITION_PCT
    if mode == "kelly":
        if avg_loss <= 0:
            return capital * SIMULATION_POSITION_PCT
        b = avg_win / avg_loss
        p = win_rate
        q = 1.0 - p
        k = (b * p - q) / b if b > 0 else 0.0
        k = max(0.0, min(k, 1.0)) * SIMULATION_KELLY_FRACTION
        if k <= 0:
            k = SIMULATION_POSITION_PCT
        return capital * k
    return min(SIMULATION_FIXED_SIZE_USDT, capital)


def _read_path(db, symbol, timeframe, entry_ts, exit_ts, entry_price,
               direction, sl_pct, tp_pct):
    """
    Проходит по свечам от entry_ts до exit_ts и проверяет касание SL/TP.
    Возвращает (actual_exit_ts, actual_exit_price, reason).
    """
    tf_ms = _tf_ms(timeframe)
    cur_ts = entry_ts
    if direction == "UP":
        sl_price = entry_price * (1.0 - sl_pct) if sl_pct > 0 else None
        tp_price = entry_price * (1.0 + tp_pct) if tp_pct > 0 else None
    else:
        sl_price = entry_price * (1.0 + sl_pct) if sl_pct > 0 else None
        tp_price = entry_price * (1.0 - tp_pct) if tp_pct > 0 else None

    while cur_ts <= exit_ts:
        rows = db.get_candles(symbol, timeframe)
        break

    candles = db.get_candles(symbol, timeframe)
    if not candles:
        return exit_ts, entry_price, "no-data"

    idx_map = {c[0]: i for i, c in enumerate(candles)}
    start_i = idx_map.get(entry_ts)
    end_i = idx_map.get(exit_ts)
    if start_i is None:
        return exit_ts, entry_price, "no-entry"
    if end_i is None:
        end_i = len(candles) - 1

    for i in range(start_i, min(end_i + 1, len(candles))):
        ts, _o, h, l, c, _v = candles[i]
        if direction == "UP":
            if sl_price is not None and l <= sl_price:
                return ts, sl_price, "sl"
            if tp_price is not None and h >= tp_price:
                return ts, tp_price, "tp"
        else:
            if sl_price is not None and h >= sl_price:
                return ts, sl_price, "sl"
            if tp_price is not None and l <= tp_price:
                return ts, tp_price, "tp"

    exit_close = db.get_candle_close_at(symbol, timeframe, exit_ts)
    if exit_close is None:
        exit_close = candles[-1][4]
    return exit_ts, exit_close, "horizon"


def run_simulation(db, symbol, timeframe,
                   initial_capital,
                   fee_pct=SIMULATION_FEE_PCT,
                   stop_loss_pct=SIMULATION_STOP_LOSS_PCT,
                   take_profit_pct=SIMULATION_TAKE_PROFIT_PCT,
                   sizing_mode=SIMULATION_SIZING_MODE,
                   only_resolved=True,
                   skip_flat=True):
    preds = db.get_predictions(symbol=symbol, timeframe=timeframe,
                               limit=100000)
    if only_resolved:
        preds = [p for p in preds if p["status"] == "RESOLVED"]
    if skip_flat:
        preds = [p for p in preds if p["direction"] in ("UP", "DOWN")]
    preds.sort(key=lambda p: p["candle_ts"])

    capital = float(initial_capital)
    peak = capital
    max_dd = 0.0
    wins = 0
    losses = 0
    trades = []
    equity_curve = [(0, capital)]

    win_rate_est = 0.5
    avg_win_est = 0.01
    avg_loss_est = 0.01
    closed = 0

    for p in preds:
        entry_ts = p["candle_ts"]
        tf_ms = _tf_ms(timeframe)
        horizon_idx = max(
            1, int(round(p["horizon_seconds"] * 1000 / tf_ms)))
        exit_ts = entry_ts + horizon_idx * tf_ms

        entry_price = db.get_candle_close_at(
            symbol, timeframe, entry_ts)
        if entry_price is None or entry_price <= 0:
            continue

        size = _position_size(sizing_mode, capital, entry_price,
                              win_rate_est, avg_win_est, avg_loss_est)
        if size <= 0:
            continue

        direction = p["direction"]

        actual_exit_ts, actual_exit_price, reason = _read_path(
            db, symbol, timeframe, entry_ts, exit_ts, entry_price,
            direction, stop_loss_pct, take_profit_pct)

        if direction == "UP":
            raw_ret = actual_exit_price / entry_price - 1.0
        else:
            raw_ret = 1.0 - actual_exit_price / entry_price

        gross = size * raw_ret
        fees = size * fee_pct * 2.0
        pnl = gross - fees
        capital += pnl

        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

        closed += 1
        if closed > 0:
            win_rate_est = wins / closed
            if wins > 0:
                avg_win_est = max(0.001,
                                  (capital - initial_capital)
                                  / max(1, wins * initial_capital))
            if losses > 0:
                avg_loss_est = max(0.001,
                                   abs((capital - initial_capital))
                                   / max(1, losses * initial_capital))

        trades.append({
            "prediction_id": p["id"],
            "entry_ts": entry_ts,
            "exit_ts": actual_exit_ts,
            "entry_price": entry_price,
            "exit_price": actual_exit_price,
            "direction": direction,
            "size": size,
            "pnl": pnl,
            "return_pct": pnl / size if size > 0 else 0.0,
            "reason": reason,
        })
        equity_curve.append((actual_exit_ts, capital))

    profit = capital - initial_capital
    return_pct = profit / initial_capital if initial_capital > 0 else 0.0
    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0
    avg_trade = profit / total if total > 0 else 0.0

    record = {
        "created_at": int(time.time()),
        "symbol": symbol,
        "timeframe": timeframe,
        "initial_capital": float(initial_capital),
        "final_capital": float(capital),
        "profit": float(profit),
        "return_pct": float(return_pct),
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": float(win_rate),
        "avg_trade": float(avg_trade),
        "max_drawdown": float(max_dd),
        "config_json": json.dumps({
            "fee_pct": fee_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "sizing_mode": sizing_mode,
            "only_resolved": only_resolved,
            "skip_flat": skip_flat,
        }),
    }

    sim_id = db.insert_simulation(record)
    rows = [
        (sim_id, t["prediction_id"], t["entry_ts"], t["exit_ts"],
         t["entry_price"], t["exit_price"], t["direction"], t["size"],
         t["pnl"], t["return_pct"])
        for t in trades
    ]
    db.insert_simulation_trades(rows)
    record["id"] = sim_id
    record["equity_curve"] = equity_curve
    return record, trades