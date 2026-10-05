# alerts.py
import threading

from config import (
    ALERTS_ENABLED_DEFAULT,
    ALERTS_PREDICTION_MADE,
    ALERTS_PREDICTION_RESOLVED,
    ALERTS_ACCURACY_DROP_PCT,
    ALERTS_WINDOW,
)


class Notifier:
    def __init__(self, root, db):
        self.root = root
        self.db = db
        self.enabled = ALERTS_ENABLED_DEFAULT
        self._tray = None
        self._recent = []

    def set_enabled(self, flag):
        self.enabled = bool(flag)

    def attach_tray(self, tray):
        self._tray = tray

    def on_prediction_made(self, payload):
        if not self.enabled or not ALERTS_PREDICTION_MADE:
            return
        d = payload.get("direction")
        p = payload.get("probability", 0.0)
        r = payload.get("expected_return", 0.0)
        sym = payload.get("symbol")
        tf = payload.get("timeframe")
        h = payload.get("horizon_seconds", 0)
        title = f"Prediction: {d} ({sym} {tf})"
        body = (f"p={p * 100:.1f}%  "
                f"E[ret]={r * 100:+.3f}%  "
                f"h={h}s")
        self._notify(title, body)

    def on_prediction_resolved(self, payload):
        if not self.enabled or not ALERTS_PREDICTION_RESOLVED:
            return
        ok = payload.get("direction_correct")
        actual = payload.get("actual_return", 0.0)
        sym = payload.get("symbol")
        tf = payload.get("timeframe")
        tag = "CORRECT" if ok else "WRONG"
        title = f"{tag}: {sym} {tf}"
        body = f"actual={actual * 100:+.2f}%"
        self._notify(title, body)

    def check_accuracy(self, symbol, timeframe):
        if not self.enabled:
            return
        rows = self.db.get_predictions(
            symbol=symbol, timeframe=timeframe, limit=ALERTS_WINDOW)
        resolved = [r for r in rows if r["status"] == "RESOLVED"]
        if len(resolved) < 10:
            return
        hits = sum(1 for r in resolved
                   if r["direction_correct"] == 1)
        acc = hits / len(resolved)
        if acc < ALERTS_ACCURACY_DROP_PCT:
            self._notify(
                "Accuracy drop",
                f"{symbol} {timeframe}: "
                f"hit-rate {acc * 100:.1f}% over "
                f"last {len(resolved)} resolved")

    def _notify(self, title, body):
        try:
            if self._tray is not None:
                self._tray.notify(title, body)
                return
        except Exception:
            pass
        try:
            if self.root is not None:
                self.root.after(0, lambda: self._fallback(title, body))
        except Exception:
            pass

    def _fallback(self, title, body):
        try:
            from tkinter import messagebox
            messagebox.showinfo(title, body)
        except Exception:
            print(f"[alert] {title} — {body}")