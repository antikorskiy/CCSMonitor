# ui.py
import time
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
from alerts import Notifier
from tray import TrayIcon

from matplotlib.dates import DateFormatter, AutoDateLocator
from matplotlib.collections import LineCollection, PolyCollection
import os
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from export import export_rows
from simulation import run_simulation
import json
from config import (
    DEFAULT_SYMBOL, DEFAULT_TIMEFRAME,
    SYMBOLS, TIMEFRAMES,
    AUTO_SYNC_DEFAULT_ENABLED, AUTO_SYNC_DEFAULT_INTERVAL,
    AUTO_SYNC_MIN_INTERVAL, AUTO_ML_DEFAULT_ENABLED,
    PREDICTION_FLAT_EPS, PREDICTIONS_HISTORY_MAX,
    HISTORY_OPTIONS, HISTORY_DEFAULT,
    EXPORT_DEFAULT_DIR, EXPORT_FORMATS,
    SIMULATION_DEFAULT_CAPITAL,
    FEATURE_SCHEMA_VERSION,
    PROXY_MODES, PROXY_DEFAULT_MODE, NETWORK_REFRESH_MS,
    DB_PATH, ML_HORIZONS, ML_DEFAULT_HORIZON,
    SIMULATION_STOP_LOSS_PCT,
    SIMULATION_TAKE_PROFIT_PCT,
    SIMULATION_SIZING_OPTIONS,
    SIMULATION_SIZING_MODE,
    SIMULATION_POSITION_PCT,
    SIMULATION_FIXED_SIZE_USDT,
    HOTKEYS, ALERTS_ENABLED_DEFAULT,
)
from network import format_bytes, format_rate
from storage import storage_report
from features import compute_indicators
from ml import FEATURE_COLS


BG = "#0e1016"
FG = "#cfd3dc"
GRID = "#242833"
CROSSHAIR = "#8a93a6"

DEFAULT_VISIBLE = 200
ZOOM_THROTTLE_MS = 16

QUOTES = ("USDT", "FDUSD", "BUSD", "USDC", "TUSD", "BTC", "ETH", "BNB")

GREEN = "#00c878"
RED = "#dc3c50"


def pretty_symbol(sym):
    for q in QUOTES:
        if sym.endswith(q) and len(sym) > len(q):
            return f"{sym[:-len(q)]} ({q})"
    return sym


def _ts_to_mpl(ts_ms):
    return ts_ms / 1000.0 / 86400.0 + 719163.0


def _clip_segment(x0, y0, x1, y1, xlim, ylim):
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0
    if dx != 0:
        ta = (xlim[0] - x0) / dx
        tb = (xlim[1] - x0) / dx
        t0 = max(t0, min(ta, tb))
        t1 = min(t1, max(ta, tb))
    if dy != 0:
        ta = (ylim[0] - y0) / dy
        tb = (ylim[1] - y0) / dy
        t0 = max(t0, min(ta, tb))
        t1 = min(t1, max(ta, tb))
    if t1 <= t0:
        return None
    return ((x0 + t0 * dx, y0 + t0 * dy),
            (x0 + t1 * dx, y0 + t1 * dy))

class FlowToolbar(tk.Frame):
    def __init__(self, master, bg, gap=6, row_gap=4,
                 min_row_height=26):
        super().__init__(master, bg=bg)
        self._gap = gap
        self._row_gap = row_gap
        self._min_h = min_row_height
        self._items = []
        self._last_width = -1
        self.pack_propagate(False)
        self.configure(height=min_row_height + 4)
        self.bind("<Configure>", self._on_configure)

    def add(self, widget):
        self._items.append(widget)
        self.after_idle(self._relayout)

    def add_separator(self):
        sep = tk.Frame(self, bg=GRID, width=1)
        self._items.append(sep)
        self.after_idle(self._relayout)
        return sep

    def _on_configure(self, event):
        if event.width == self._last_width:
            return
        self._last_width = event.width
        self._relayout()

    def _relayout(self):
        w = self.winfo_width()
        if w < 50:
            return

        x = 0
        y = 0
        row_h = 0

        for item in self._items:
            iw = item.winfo_reqwidth()
            ih = item.winfo_reqheight()

            is_sep = (isinstance(item, tk.Frame)
                      and int(item.cget("width") or 0) == 1)
            if is_sep:
                ih = self._min_h
            if ih < self._min_h:
                ih = self._min_h

            if x > 0 and x + iw > w:
                x = 0
                y += row_h + self._row_gap
                row_h = 0

            item.place(x=x, y=y, width=iw, height=ih)
            x += iw + self._gap
            if ih > row_h:
                row_h = ih

        self.configure(height=y + row_h + 4)

class CCSMonitorApp:
    def __init__(self, root, db, engine):
        self.root = root
        self.db = db
        self.engine = engine
        self.engine.subscribe(self._engine_event)

        self._tray = None
        self._alerts_enabled = ALERTS_ENABLED_DEFAULT
        self._notifier = None

        self._symbol = None
        self._timeframe = None

        self._x = None
        self._o = None
        self._h = None
        self._l = None
        self._c = None
        self._v = None

        self._ema_fast = None
        self._ema_slow = None
        self._rsi = None

        self._n = 0
        self._candles = None

        self._hover_ann = None
        self._vline_price = None
        self._vline_rsi = None
        self._hline_price = None
        self._hover_visible = False
        self._hover_idx = -1

        self._bg_price = None
        self._bg_rsi = None
        self._blit_ready = False

        self._zoom_pending = None
        self._zoom_center = None
        self._zoom_scheduled = None

        self._prediction = None
        self._pending_highlight_ts = None
        self._genome_tree_map = {}

        self._pipeline_state = {
            "api": "pending",
            "dataset": "pending",
            "validation": "pending",
            "features": "pending",
            "genome": "pending",
            "ml": "pending",
            "prediction": "pending",
            "live": "pending",
        }
        self._pipeline_win = None
        self._pipeline_labels = {}

        self._last_sync_str = "—"
        self._auto_started_for = None

        self.root.title("CCSMonitor")
        self.root.geometry("1400x900")
        self.root.configure(bg=BG)

        self._build_toolbar()
        self._build_figure(root)
        self._build_nav(root)

        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("axes_leave_event", self._on_leave)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self.statusbar = tk.Label(root, text="Ready", bg=BG, fg="#8a93a6",
                                  anchor="w", padx=10)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self.on_load)

        self._setup_hotkeys()
        self._setup_tray()

    def _build_toolbar(self):
        container = tk.Frame(self.root, bg=BG)
        container.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 2))

        self.lbl_status = tk.Label(container, text="Ready", bg=BG,
                                   fg="#8a93a6")
        self.lbl_status.pack(side=tk.RIGHT, padx=(8, 0))

        self._toolbar = FlowToolbar(container, bg=BG)
        self._toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tb = self._toolbar

        # --- Asset ---
        lab = tk.Label(tb, text="Asset:", bg=BG, fg=FG)
        tb.add(lab)

        self._pretty_to_sym = {pretty_symbol(s): s for s in SYMBOLS}
        self.symbol_var = tk.StringVar(value=pretty_symbol(DEFAULT_SYMBOL))
        self.cb_symbol = ttk.Combobox(
            tb, textvariable=self.symbol_var,
            values=list(self._pretty_to_sym.keys()),
            state="readonly", width=14)
        tb.add(self.cb_symbol)
        self.cb_symbol.bind("<<ComboboxSelected>>", lambda e: self.on_load())

        # --- TF ---
        lab = tk.Label(tb, text="TF:", bg=BG, fg=FG)
        tb.add(lab)

        self.tf_var = tk.StringVar(value=DEFAULT_TIMEFRAME)
        self.cb_tf = ttk.Combobox(
            tb, textvariable=self.tf_var,
            values=TIMEFRAMES, state="readonly", width=5)
        tb.add(self.cb_tf)
        self.cb_tf.bind("<<ComboboxSelected>>", lambda e: self.on_load())

        # --- History ---
        lab = tk.Label(tb, text="History:", bg=BG, fg=FG)
        tb.add(lab)

        self.history_var = tk.StringVar(value=HISTORY_DEFAULT)
        self.cb_history = ttk.Combobox(
            tb, textvariable=self.history_var,
            values=HISTORY_OPTIONS, state="readonly", width=6)
        tb.add(self.cb_history)
        self.cb_history.bind("<<ComboboxSelected>>",
                             lambda e: self._on_history_change())
        self.engine.set_history_years(HISTORY_DEFAULT)

        # --- LOAD ---
        self.btn_load = ttk.Button(tb, text="LOAD / SYNC",
                                   command=self.on_load)
        tb.add(self.btn_load)

        tb.add_separator()

        # --- Zoom ---
        b = ttk.Button(tb, text="+", width=3,
                       command=lambda: self._zoom(0.8))
        tb.add(b)
        b = ttk.Button(tb, text="−", width=3,
                       command=lambda: self._zoom(1.25))
        tb.add(b)
        b = ttk.Button(tb, text="Reset", command=self._reset_zoom)
        tb.add(b)

        tb.add_separator()

        # --- Auto-sync ---
        self.auto_var = tk.BooleanVar(value=AUTO_SYNC_DEFAULT_ENABLED)
        b = ttk.Checkbutton(tb, text="Auto-sync", variable=self.auto_var,
                            command=self._on_auto_toggle)
        tb.add(b)

        lab = tk.Label(tb, text="every", bg=BG, fg=FG)
        tb.add(lab)

        self.interval_var = tk.StringVar(value=str(AUTO_SYNC_DEFAULT_INTERVAL))
        self.interval_entry = ttk.Entry(tb, textvariable=self.interval_var,
                                        width=5)
        tb.add(self.interval_entry)

        lab = tk.Label(tb, text="s", bg=BG, fg=FG)
        tb.add(lab)

        self.interval_entry.bind("<FocusOut>",
                                 lambda e: self._on_auto_toggle())
        self.interval_entry.bind("<Return>",
                                 lambda e: self._on_auto_toggle())

        # --- Sync now ---
        self.btn_sync_now = ttk.Button(tb, text="Sync now",
                                       command=self._on_sync_now)
        tb.add(self.btn_sync_now)

        tb.add_separator()

        # --- Auto ML ---
        self.auto_ml_var = tk.BooleanVar(value=AUTO_ML_DEFAULT_ENABLED)
        b = ttk.Checkbutton(tb, text="Auto ML", variable=self.auto_ml_var,
                            command=self._on_auto_ml_toggle)
        tb.add(b)
        self.engine.set_auto_ml_enabled(AUTO_ML_DEFAULT_ENABLED)

        tb.add_separator()

        # --- ML ---
        self.btn_train = ttk.Button(tb, text="Train ML",
                                    command=self._on_train_ml)
        tb.add(self.btn_train)

        self.btn_predict = ttk.Button(tb, text="Predict",
                                      command=self._on_predict)
        tb.add(self.btn_predict)

        self.btn_preds = ttk.Button(tb, text="Predictions",
                                    command=self._open_predictions_window)
        tb.add(self.btn_preds)

        tb.add_separator()

        # --- Genome ---
        self.btn_genome_tree = ttk.Button(tb, text="Genomes",
                                          command=self._open_genomes_window)
        tb.add(self.btn_genome_tree)

        self.btn_genomes = ttk.Button(tb, text="Rebuild Genomes",
                                      command=self._on_rebuild_genomes)
        tb.add(self.btn_genomes)

        tb.add_separator()

        # --- Logs / Sim / Export ---
        self.btn_logs = ttk.Button(tb, text="Logs",
                                   command=self._open_logs_window)
        tb.add(self.btn_logs)

        self.btn_sim = ttk.Button(tb, text="Simulate",
                                  command=self._open_simulation_window)
        tb.add(self.btn_sim)

        self.btn_export = ttk.Button(tb, text="Export",
                                     command=self._open_export_window)
        tb.add(self.btn_export)

        tb.add_separator()

        # --- Network / Proxy ---
        self.btn_network = ttk.Button(tb, text="Network",
                                      command=self._open_network_window)
        tb.add(self.btn_network)

        self.btn_proxy = ttk.Button(tb, text="Proxy",
                                    command=self._open_proxy_window)
        tb.add(self.btn_proxy)

        tb.add_separator()

        # --- Infrastructure ---
        self.btn_pipeline = ttk.Button(tb, text="Pipeline",
                                       command=self._open_pipeline_window)
        tb.add(self.btn_pipeline)

        self.btn_coverage = ttk.Button(tb, text="Coverage",
                                       command=self._open_coverage_window)
        tb.add(self.btn_coverage)

        self.btn_database = ttk.Button(tb, text="Database",
                                       command=self._open_database_window)
        tb.add(self.btn_database)

        tb.add_separator()

        # --- ML insights ---
        self.btn_ml_insights = ttk.Button(tb, text="ML Insights",
                                          command=self._open_ml_insights_window)
        tb.add(self.btn_ml_insights)

        self.btn_train_all = ttk.Button(tb, text="Train All",
                                        command=self._on_train_all)
        tb.add(self.btn_train_all)

        self.btn_predict_all = ttk.Button(tb, text="Predict All",
                                          command=self._on_predict_all)
        tb.add(self.btn_predict_all)

        self.btn_consensus = ttk.Button(tb, text="Consensus",
                                        command=self._open_consensus_window)
        tb.add(self.btn_consensus)

    def _build_figure(self, root):
        self.fig = Figure(figsize=(12, 8), dpi=100, facecolor=BG)
        self.ax_price = self.fig.add_subplot(211)
        self.ax_rsi = self.fig.add_subplot(212, sharex=self.ax_price)
        self._style_axes()
        self.fig.subplots_adjust(left=0.06, right=0.985,
                                 top=0.955, bottom=0.07, hspace=0.08)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                         expand=True)

    def _build_nav(self, root):
        nav_frame = tk.Frame(root, bg=BG)
        nav_frame.pack(side=tk.TOP, fill=tk.X)
        self.nav = NavigationToolbar2Tk(self.canvas, nav_frame,
                                        pack_toolbar=False)
        self.nav.update()
        self.nav.pack(side=tk.LEFT)
        self.nav.configure(bg=BG)
        for child in self.nav.winfo_children():
            try:
                child.configure(bg=BG)
            except tk.TclError:
                pass

    def _style_axes(self):
        for ax in (self.ax_price, self.ax_rsi):
            ax.set_facecolor(BG)
            ax.tick_params(colors=FG, labelsize=9)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
            ax.yaxis.label.set_color(FG)

        self.ax_price.set_ylabel("Price", color=FG)
        self.ax_rsi.set_ylabel("RSI", color=FG)
        self.ax_rsi.set_ylim(0, 100)
        self.ax_rsi.axhline(70, color="#444a5a",
                            linestyle="--", linewidth=0.7)
        self.ax_rsi.axhline(30, color="#444a5a",
                            linestyle="--", linewidth=0.7)

        self.ax_rsi.xaxis.set_major_locator(AutoDateLocator())
        self.ax_rsi.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
        self.ax_price.tick_params(labelbottom=False)

    def _engine_event(self, event, payload):
        self.root.after(0, self._handle_engine_event, event, payload)

    def _handle_engine_event(self, event, payload):
        self._pipeline_event(event, payload)
        self._alert_event(event, payload)
        if event == "sync_started":
            sym = pretty_symbol(payload["symbol"])
            self.lbl_status.config(
                text=f"Syncing {sym} {payload['timeframe']}...")

        elif event == "sync_error":
            self.lbl_status.config(text="Sync error")
            self.statusbar.config(text=f"Sync error: {payload['error']}")

        elif event == "candles_updated":
            self._on_candles_updated(
                payload["symbol"], payload["timeframe"],
                payload["candles"], payload["inserted"],
                payload.get("features_added", 0),
                payload.get("genomes_added", 0),
                payload.get("instances_added", 0))

        elif event == "auto_train_started":
            sym = pretty_symbol(payload["symbol"])
            self.lbl_status.config(
                text=f"Auto-train {sym} {payload['timeframe']} "
                     f"(+{payload['new_candles']} candles)...")

        elif event == "model_trained":
            self._on_model_trained(**payload)

        elif event == "prediction_made":
            self._on_prediction_made(**payload)

        elif event == "prediction_resolved":
            self._on_prediction_resolved(**payload)

        elif event == "history_started":
            sym = pretty_symbol(payload["symbol"])
            self.lbl_status.config(
                text=f"Downloading history {sym} "
                     f"{payload['timeframe']}...")

        elif event == "history_progress":
            sym = pretty_symbol(payload["symbol"])
            self.lbl_status.config(
                text=f"History {sym} {payload['timeframe']}: "
                     f"page {payload['page']} · {payload['count']} candles")

        elif event == "history_finished":
            sym = pretty_symbol(payload["symbol"])
            self.lbl_status.config(
                text=f"History {sym} {payload['timeframe']}: "
                     f"{payload['total']} candles downloaded")

    def _pipeline_event(self, event, payload):
        mapping = {
            "sync_started": [("api", "running")],
            "candles_updated": [("api", "done"),
                                ("dataset", "done"),
                                ("features", "done"),
                                ("validation", "done")],
            "history_started": [("dataset", "running")],
            "history_finished": [("dataset", "done")],
            "sync_error": [("api", "error")],
            "auto_train_started": [("ml", "running")],
            "model_trained": [("ml", "done")],
            "prediction_made": [("prediction", "done"),
                                ("live", "done")],
        }
        for stage, status in mapping.get(event, []):
            self._pipeline_state[stage] = status
        self._refresh_pipeline_labels()

    def _alert_event(self, event, payload):
        if self._notifier is None:
            return
        if event == "prediction_made":
            self._notifier.on_prediction_made(payload)
        elif event == "prediction_resolved":
            self._notifier.on_prediction_resolved(payload)
            sym = payload.get("symbol")
            tf = payload.get("timeframe")
            if sym and tf:
                self._notifier.check_accuracy(sym, tf)
                
    def _setup_hotkeys(self):
        for action, seq in HOTKEYS.items():
            fn = self._hotkey_action(action)
            if fn is None:
                continue
            try:
                self.root.bind_all(seq, lambda e, f=fn: f())
            except Exception:
                pass

    def _hotkey_action(self, action):
        table = {
            "load": self.on_load,
            "sync_now": self._on_sync_now,
            "train": self._on_train_ml,
            "predict": self._on_predict,
            "predict_all": self._on_predict_all,
            "consensus": self._open_consensus_window,
            "genomes": self._open_genomes_window,
            "insights": self._open_ml_insights_window,
            "simulate": self._open_simulation_window,
            "export": self._open_export_window,
            "reset_zoom": self._reset_zoom,
            "zoom_in": lambda: self._zoom(0.8),
            "zoom_out": lambda: self._zoom(1.25),
            "network": self._open_network_window,
            "proxy": self._open_proxy_window,
            "pipeline": self._open_pipeline_window,
            "coverage": self._open_coverage_window,
            "database": self._open_database_window,
            "logs": self._open_logs_window,
            "close": self._close_top_window,
        }
        return table.get(action)

    def _close_top_window(self):
        for attr in ("_sim_win", "_preds_win", "_logs_win",
                     "_genomes_win", "_net_win", "_proxy_win",
                     "_pipeline_win", "_coverage_win",
                     "_db_win", "_ml_win", "_cons_win"):
            win = getattr(self, attr, None)
            if win is not None and win.winfo_exists():
                win.destroy()
                return

    def _setup_tray(self):
        self._notifier = Notifier(self.root, self.db)
        self._notifier.enabled = ALERTS_ENABLED_DEFAULT

        self._tray = TrayIcon(
            on_show=self._tray_show,
            on_quit=self._tray_quit,
            on_sync=self._on_sync_now,
            on_train=self._on_train_ml,
            on_predict=self._on_predict,
            on_toggle_alerts=self._tray_toggle_alerts,
        )
        if self._tray.available():
            self._tray.start()
            self._notifier.attach_tray(self._tray)

    def _tray_show(self):
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass

    def _tray_quit(self):
        try:
            self.root.after(0, self._on_close)
        except Exception:
            pass

    def _tray_toggle_alerts(self, flag):
        self._alerts_enabled = bool(flag)
        if self._notifier is not None:
            self._notifier.set_enabled(flag)
        try:
            self.root.after(0, lambda: self.statusbar.config(
                text=f"Alerts: {'ON' if flag else 'OFF'}"))
        except Exception:
            pass

    def _shutdown_tray(self):
        if self._tray is not None:
            self._tray.stop()

    def _open_pipeline_window(self):
        if self._pipeline_win is not None \
                and self._pipeline_win.winfo_exists():
            self._pipeline_win.lift()
            self._refresh_pipeline_labels()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Pipeline")
        win.geometry("420x420")
        win.configure(bg=BG)
        self._pipeline_win = win

        tk.Label(win, text="Pipeline stages", bg=BG, fg=FG,
                 font=("Consolas", 11, "bold")
                 ).pack(anchor="w", padx=16, pady=(16, 8))

        self._pipeline_labels = {}
        stages = [
            ("api", "API Connection"),
            ("dataset", "Dataset"),
            ("validation", "Validation"),
            ("features", "Feature Engine"),
            ("genome", "Genome Engine"),
            ("ml", "ML"),
            ("prediction", "Prediction"),
            ("live", "Live Monitoring"),
        ]
        for key, label in stages:
            row = tk.Frame(win, bg=BG)
            row.pack(side=tk.TOP, fill=tk.X, padx=16, pady=2)
            dot = tk.Label(row, text="●", bg=BG, fg="#8a93a6",
                           font=("Consolas", 12), width=2)
            dot.pack(side=tk.LEFT)
            name = tk.Label(row, text=label, bg=BG, fg=FG,
                            anchor="w", font=("Consolas", 10))
            name.pack(side=tk.LEFT)
            status = tk.Label(row, text="pending", bg=BG, fg="#8a93a6",
                              anchor="e", font=("Consolas", 9))
            status.pack(side=tk.RIGHT)
            self._pipeline_labels[key] = (dot, status)

        self._refresh_pipeline_labels()

    def _refresh_pipeline_labels(self):
        if not self._pipeline_labels:
            return
        colors = {
            "pending": "#8a93a6",
            "running": "#ffb74d",
            "done": GREEN,
            "error": RED,
        }
        for key, (dot, status) in self._pipeline_labels.items():
            st = self._pipeline_state.get(key, "pending")
            dot.config(fg=colors.get(st, "#8a93a6"))
            status.config(text=st, fg=colors.get(st, "#8a93a6"))

    def _open_coverage_window(self):
        if getattr(self, "_coverage_win", None) is not None \
                and self._coverage_win.winfo_exists():
            self._coverage_win.lift()
            self._refresh_coverage_window()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Dataset Coverage")
        win.geometry("1000x560")
        win.configure(bg=BG)
        self._coverage_win = win

        head = tk.Frame(win, bg=BG)
        head.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        tk.Label(head, text="Coverage", bg=BG, fg=FG,
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT)
        ttk.Button(head, text="Refresh",
                   command=self._refresh_coverage_window
                   ).pack(side=tk.RIGHT)
        ttk.Button(head, text="Download missing",
                   command=self._download_missing_gaps
                   ).pack(side=tk.RIGHT, padx=(0, 8))

        cols = ("symbol", "tf", "count", "first", "last",
                "years", "gaps", "missing")
        tv = ttk.Treeview(win, columns=cols, show="headings", height=12)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=110, anchor="center")
        tv.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                padx=8, pady=(0, 6))
        tv.bind("<<TreeviewSelect>>",
                lambda e: self._refresh_coverage_gaps())
        self._coverage_tv = tv

        tk.Label(win, text="Gaps (selected row)", bg=BG, fg=FG,
                 anchor="w", font=("Consolas", 10, "bold")
                 ).pack(anchor="w", padx=8)
        gcols = ("from", "to", "missing")
        gtv = ttk.Treeview(win, columns=gcols, show="headings", height=8)
        for c in gcols:
            gtv.heading(c, text=c)
            gtv.column(c, width=180, anchor="center")
        gtv.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                 padx=8, pady=(0, 8))
        self._coverage_gaps_tv = gtv

        self._refresh_coverage_window()

    def _refresh_coverage_window(self):
        tv = getattr(self, "_coverage_tv", None)
        if tv is None:
            return
        for i in tv.get_children():
            tv.delete(i)

        rows = self.db.coverage_summary()
        for r in rows:
            years = self._years_span(r["first"], r["last"])
            gaps = self.db.find_gaps(r["symbol"], r["timeframe"], limit=5000)
            missing = sum(g["missing"] for g in gaps)
            tv.insert("", "end", values=(
                pretty_symbol(r["symbol"]),
                r["timeframe"],
                r["count"],
                self._fmt_ts(r["first"]),
                self._fmt_ts(r["last"]),
                years,
                len(gaps),
                missing,
            ))

    def _refresh_coverage_gaps(self):
        tv = getattr(self, "_coverage_tv", None)
        gtv = getattr(self, "_coverage_gaps_tv", None)
        if tv is None or gtv is None:
            return
        for i in gtv.get_children():
            gtv.delete(i)
        sel = tv.selection()
        if not sel:
            return
        vals = tv.item(sel[0], "values")
        if not vals:
            return
        symbol = self._pretty_to_sym.get(vals[0], vals[0])
        timeframe = vals[1]
        gaps = self.db.find_gaps(symbol, timeframe, limit=500)
        for g in gaps:
            gtv.insert("", "end", values=(
                self._fmt_ts(g["from_ts"]),
                self._fmt_ts(g["to_ts"]),
                g["missing"],
            ))

    def _download_missing_gaps(self):
        tv = getattr(self, "_coverage_tv", None)
        if tv is None:
            return
        sel = tv.selection()
        if not sel:
            return
        vals = tv.item(sel[0], "values")
        if not vals:
            return
        symbol = self._pretty_to_sym.get(vals[0], vals[0])
        timeframe = vals[1]
        self.statusbar.config(
            text=f"Downloading gaps for {pretty_symbol(symbol)} {timeframe}...")
        self.engine.request_fill_gaps(symbol, timeframe)

    @staticmethod
    def _years_span(first_ms, last_ms):
        if not first_ms or not last_ms:
            return "—"
        days = (last_ms - first_ms) / 86400000.0
        if days < 30:
            return f"{days:.1f}d"
        if days < 365:
            return f"{days / 30.44:.1f}mo"
        return f"{days / 365.25:.1f}y"

    def _open_database_window(self):
        if getattr(self, "_db_win", None) is not None \
                and self._db_win.winfo_exists():
            self._db_win.lift()
            self._refresh_database_window()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Database")
        win.geometry("1000x640")
        win.configure(bg=BG)
        self._db_win = win

        head = tk.Frame(win, bg=BG)
        head.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        tk.Label(head, text="Database", bg=BG, fg=FG,
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT)
        ttk.Button(head, text="Refresh",
                   command=self._refresh_database_window
                   ).pack(side=tk.RIGHT)

        nb = ttk.Notebook(win)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                padx=8, pady=(0, 8))
        self._db_nb = nb

        # ---- Overview ----
        ov = tk.Frame(nb, bg=BG)
        nb.add(ov, text="Overview")
        self._db_overview_lbl = tk.Label(
            ov, text="—", bg=BG, fg=FG, anchor="nw", justify=tk.LEFT,
            font=("Consolas", 10))
        self._db_overview_lbl.pack(side=tk.TOP, fill=tk.BOTH,
                                    expand=True, padx=12, pady=12)

        # ---- Tables ----
        tb = tk.Frame(nb, bg=BG)
        nb.add(tb, text="Tables")
        tcols = ("name", "rows", "size")
        ttv = ttk.Treeview(tb, columns=tcols,
                           show="headings", height=22)
        for c in tcols:
            ttv.heading(c, text=c)
            ttv.column(c, width=200, anchor="center")
        ttv.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                 padx=8, pady=8)
        self._db_tables_tv = ttv

        # ---- Health ----
        h = tk.Frame(nb, bg=BG)
        nb.add(h, text="Health")
        hbox = tk.Frame(h, bg=BG)
        hbox.pack(side=tk.TOP, fill=tk.X, padx=12, pady=12)
        ttk.Button(hbox, text="Integrity Check",
                   command=self._db_integrity).pack(side=tk.LEFT)
        ttk.Button(hbox, text="Quick Check",
                   command=self._db_quick_check
                   ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(hbox, text="Find Duplicates",
                   command=self._db_duplicates
                   ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(hbox, text="Find Orphans",
                   command=self._db_orphans
                   ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(hbox, text="VACUUM",
                   command=self._db_vacuum).pack(side=tk.LEFT, padx=(6, 0))

        self._db_health_lbl = tk.Label(
            h, text="Ready", bg=BG, fg=FG, anchor="nw",
            justify=tk.LEFT, font=("Consolas", 10))
        self._db_health_lbl.pack(side=tk.TOP, fill=tk.BOTH,
                                  expand=True, padx=12, pady=(0, 12))

        # ---- Storage ----
        st = tk.Frame(nb, bg=BG)
        nb.add(st, text="Storage")
        self._db_storage_lbl = tk.Label(
            st, text="—", bg=BG, fg=FG, anchor="nw", justify=tk.LEFT,
            font=("Consolas", 10))
        self._db_storage_lbl.pack(side=tk.TOP, fill=tk.BOTH,
                                   expand=True, padx=12, pady=12)

        self._refresh_database_window()

    def _refresh_database_window(self):
        ov = self.db.overview()
        oldest = self._fmt_ts(ov["oldest"]) if ov["oldest"] else "—"
        newest = self._fmt_ts(ov["newest"]) if ov["newest"] else "—"
        db_size = storage_report(DB_PATH, models_dir="models",
                                 logs_dir="logs",
                                 exports_dir="exports")

        lines = [
            f"Database size:   {format_bytes(db_size['database'])}",
            f"Tables:          {ov['tables']}",
            f"Total rows:      {ov['total_rows']}",
            "",
            f"candles:           {ov['counts']['candles']}",
            f"features:          {ov['counts']['features']}",
            f"predictions:       {ov['counts']['predictions']}",
            f"  pending:         {ov['pending']}",
            f"  resolved:        {ov['resolved']}",
            f"genomes:           {ov['counts']['genomes']}",
            f"genome_instances:  {ov['counts']['genome_instances']}",
            f"models:            {ov['counts']['models']}",
            f"parse_runs:        {ov['counts']['parse_runs']}",
            f"history_requests:  {ov['counts']['history_requests']}",
            f"simulations:       {ov['counts']['simulations']}",
            "",
            f"Oldest:          {oldest}",
            f"Newest:          {newest}",
        ]
        self._db_overview_lbl.config(text="\n".join(lines))

        # Tables
        tv = self._db_tables_tv
        for i in tv.get_children():
            tv.delete(i)
        sizes = self.db.table_sizes()
        for row in self.db.table_stats():
            size_bytes = sizes.get(row["name"])
            size_s = format_bytes(size_bytes) if size_bytes else "—"
            tv.insert("", "end", values=(
                row["name"], row["rows"], size_s))

        # Storage
        total = db_size["total"]
        slines = [
            f"Database:   {format_bytes(db_size['database'])}",
            f"Models:     {format_bytes(db_size['models'])}  "
            f"({db_size['models_count']} files)",
            f"Logs:       {format_bytes(db_size['logs'])}",
            f"Exports:    {format_bytes(db_size['exports'])}",
            "",
            f"Total:      {format_bytes(total)}",
            "",
            f"DB stat:    "
            f"{'available' if self.db.has_dbstat() else 'not available'}",
        ]
        self._db_storage_lbl.config(text="\n".join(slines))

    def _db_integrity(self):
        result = self.db.integrity_check()
        self._db_health_lbl.config(
            text=f"integrity_check: {result}",
            fg=GREEN if result.lower() == "ok" else RED)

    def _db_quick_check(self):
        result = self.db.quick_check()
        self._db_health_lbl.config(
            text=f"quick_check: {result}",
            fg=GREEN if result.lower() == "ok" else RED)

    def _db_duplicates(self):
        rows = self.db.find_duplicate_candles(limit=50)
        if not rows:
            self._db_health_lbl.config(
                text="Duplicates: none", fg=GREEN)
            return
        lines = [f"Duplicates: {len(rows)}"]
        for r in rows[:20]:
            lines.append(
                f"  {pretty_symbol(r['symbol'])} {r['timeframe']} "
                f"{self._fmt_ts(r['timestamp'])} × {r['count']}")
        self._db_health_lbl.config(
            text="\n".join(lines), fg=RED)

    def _db_orphans(self):
        o = self.db.find_orphans()
        total = sum(o.values())
        if total == 0:
            self._db_health_lbl.config(text="Orphans: none", fg=GREEN)
            return
        lines = ["Orphans:"]
        for k, v in o.items():
            lines.append(f"  {k}: {v}")
        self._db_health_lbl.config(
            text="\n".join(lines), fg=RED)

    def _db_vacuum(self):
        self._db_health_lbl.config(text="VACUUM running...")
        self.root.update_idletasks()
        ok, err = self.db.vacuum()
        if ok:
            self._db_health_lbl.config(
                text="VACUUM: done", fg=GREEN)
            self._refresh_database_window()
        else:
            self._db_health_lbl.config(
                text=f"VACUUM: error {err}", fg=RED)

    def _current_symbol(self):
        return self._pretty_to_sym.get(self.symbol_var.get(), DEFAULT_SYMBOL)

    def _quote_currency(self):
        if self._symbol is None:
            return "USDT"
        for q in QUOTES:
            if self._symbol.endswith(q):
                return q
        return "USDT"

    def _horizon_idx(self, timeframe, horizon_seconds):
        unit = timeframe[-1]
        n = int(timeframe[:-1])
        tf_ms = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit] * n
        return max(1, int(round(horizon_seconds * 1000 / tf_ms)))

    def on_load(self):
        symbol = self._current_symbol()
        timeframe = self.tf_var.get()

        self.engine.stop_auto_sync_threadsafe()
        self._auto_started_for = None

        self.btn_load.config(state="disabled")
        self.lbl_status.config(
            text=f"Loading {pretty_symbol(symbol)} {timeframe}...")

        self._symbol = symbol
        self._timeframe = timeframe
        self._x = None
        self._prediction = None
        self.engine.request_sync(symbol, timeframe)

    def _on_history_change(self):
        self.engine.set_history_years(self.history_var.get())
        self.statusbar.config(
            text=f"History window: {self.history_var.get()}")

    def _on_sync_now(self):
        if self._symbol is None:
            return
        self.engine.request_sync(self._symbol, self._timeframe)

    def _on_auto_ml_toggle(self):
        on = self.auto_ml_var.get()
        self.engine.set_auto_ml_enabled(on)
        self.statusbar.config(text=f"Auto ML: {'ON' if on else 'OFF'}")

    def _on_train_ml(self):
        if self._symbol is None:
            return
        self.btn_train.config(state="disabled")
        self.lbl_status.config(text="Training ML...")
        self.engine.request_train(self._symbol, self._timeframe)

    def _on_train_all(self):
        if self._symbol is None:
            return
        self.btn_train_all.config(state="disabled")
        self.lbl_status.config(text="Training all horizons...")
        self.engine.request_train_all_horizons(
            self._symbol, self._timeframe)

    def _on_predict_all(self):
        if self._symbol is None:
            return
        self.lbl_status.config(text="Predicting all horizons...")
        self.engine.request_predict_all_horizons(
            self._symbol, self._timeframe)

    def _on_predict(self):
        if self._symbol is None:
            return
        self.engine.request_predict(self._symbol, self._timeframe)

    def _on_rebuild_genomes(self):
        if self._symbol is None:
            return
        self.btn_genomes.config(state="disabled")
        self.lbl_status.config(text="Rebuilding genomes...")
        self.engine.request_rebuild_genomes(self._symbol, self._timeframe)

    def _on_model_trained(self, **r):
        self.btn_train.config(state="normal")
        acc = r.get("cls_accuracy")
        mae = r.get("reg_mae")
        acc_s = f"{acc * 100:.2f}%" if acc is not None else "—"
        mae_s = f"{mae * 100:.3f}%" if mae is not None else "—"
        tag = "Auto-trained" if not r.get("manual", True) else "Trained"
        self.lbl_status.config(text=tag)
        self.statusbar.config(
            text=f"{tag} · n={r['n_train']} · "
                 f"acc={acc_s} · MAE={mae_s} · v={r['version']}")

    def _on_prediction_made(self, **p):
        self._prediction = p
        direction = p["direction"]
        prob = p["probability"]
        exp_ret = p["expected_return"]
        quote = self._quote_currency()
        tag = "Auto-pred" if p.get("auto") else "Prediction"

        if (p.get("symbol") == self._symbol and
                p.get("timeframe") == self._timeframe and
                self._x is not None):
            self._draw_chart(self._symbol, self._timeframe,
                             self._candles, self.ax_price.get_xlim())

        self.statusbar.config(
            text=f"{tag} · {direction} · p={prob * 100:.1f}% · "
                 f"E[ret]={exp_ret * 100:+.3f}% · "
                 f"@ {p['close']:.2f} {quote}")

    def _on_prediction_resolved(self, **payload):
        if getattr(self, "_preds_win", None) is not None \
                and self._preds_win.winfo_exists():
            self._refresh_predictions_window()
        if self._x is not None:
            self._draw_chart(self._symbol, self._timeframe,
                             self._candles, self.ax_price.get_xlim())

    def _on_auto_toggle(self):
        enabled = self.auto_var.get()
        try:
            interval = float(self.interval_var.get())
        except ValueError:
            interval = AUTO_SYNC_DEFAULT_INTERVAL
        interval = max(interval, AUTO_SYNC_MIN_INTERVAL)
        self.interval_var.set(str(int(interval)))

        if not enabled:
            self.engine.stop_auto_sync_threadsafe()
            self._auto_started_for = None
            self.statusbar.config(text="Auto-sync OFF")
            return

        if self._symbol is None:
            return

        key = (self._symbol, self._timeframe, int(interval))
        if key == self._auto_started_for:
            return
        self.engine.start_auto_sync_threadsafe(
            self._symbol, self._timeframe, interval)
        self._auto_started_for = key
        self.statusbar.config(text=f"Auto-sync ON every {int(interval)}s")

    def _on_close(self):
        if self._zoom_scheduled is not None:
            self.root.after_cancel(self._zoom_scheduled)
        self._shutdown_tray()
        self.engine.stop_auto_sync_threadsafe()
        self.root.after(100, self.root.destroy)

    def _on_candles_updated(self, symbol, timeframe, candles,
                            inserted, features_added=0,
                            genomes_added=0, instances_added=0):
        self.btn_load.config(state="normal")
        if not candles:
            self.lbl_status.config(text="No data")
            return

        same_view = (symbol == self._symbol and
                     timeframe == self._timeframe and
                     self._x is not None)

        if not same_view:
            self._symbol = symbol
            self._timeframe = timeframe
            old_xlim = None
        else:
            old_xlim = self.ax_price.get_xlim()

        if same_view and inserted == 0:
            self._last_sync_str = time.strftime("%H:%M:%S")
            self.lbl_status.config(
                text=f"{pretty_symbol(symbol)} {timeframe} · no new candles")
            self._update_statusbar()
            return

        self._draw_chart(symbol, timeframe, candles, old_xlim)
        self._last_sync_str = time.strftime("%H:%M:%S")
        self.lbl_status.config(
            text=f"{pretty_symbol(symbol)} {timeframe}: "
                 f"{len(candles)} candles (+{inserted}), "
                 f"features +{features_added}, "
                 f"genomes +{genomes_added}, "
                 f"inst +{instances_added}")
        self.btn_genomes.config(state="normal")

        if self.auto_var.get():
            self._on_auto_toggle()

    def _update_statusbar(self):
        quote = self._quote_currency()
        last_close = self._c[-1] if self._c is not None and self._n else 0.0
        g_count = self.db.count_genomes(self._symbol) \
            if self._symbol else 0
        i_count = self.db.count_genome_instances(
            self._symbol, self._timeframe) if self._symbol else 0
        self.statusbar.config(
            text=f"Last close: {last_close:.2f} {quote}   |   "
                 f"Genomes: {g_count}  Instances: {i_count}   |   "
                 f"Last sync: {self._last_sync_str}   |   "
                 f"Auto-sync: {'ON' if self.auto_var.get() else 'OFF'}")

    def _prediction_progress(self):
        p = self._prediction
        if not p or self._n == 0:
            return None
        if p.get("symbol") != self._symbol \
                or p.get("timeframe") != self._timeframe:
            return None

        src_idx = p["idx"]
        if src_idx < 0 or src_idx >= self._n:
            return None

        horizon_idx = self._horizon_idx(
            self._timeframe, p.get("horizon_seconds", 3600))
        end_idx = src_idx + horizon_idx
        cur_idx = self._n - 1

        src_close = float(self._c[src_idx])
        if src_close <= 0:
            return None
        cur_close = float(self._c[cur_idx])
        current_return = cur_close / src_close - 1.0

        expected_return = float(p["expected_return"])
        direction = p["direction"]

        if cur_idx >= end_idx:
            actual_dir = ("UP" if current_return > PREDICTION_FLAT_EPS
                          else "DOWN"
                          if current_return < -PREDICTION_FLAT_EPS
                          else "FLAT")
            correct = (actual_dir == direction)
            status = "resolved_correct" if correct else "resolved_wrong"
            remaining = 0
        else:
            remaining = end_idx - cur_idx
            exp_sign = (1 if expected_return > PREDICTION_FLAT_EPS
                        else -1 if expected_return < -PREDICTION_FLAT_EPS
                        else 0)
            cur_sign = (1 if current_return > PREDICTION_FLAT_EPS
                        else -1 if current_return < -PREDICTION_FLAT_EPS
                        else 0)

            if exp_sign == 0:
                status = ("on_track"
                          if abs(current_return) < PREDICTION_FLAT_EPS
                          else "against")
            elif cur_sign == 0:
                status = "waiting"
            elif cur_sign == exp_sign:
                if abs(current_return) >= abs(expected_return):
                    status = "exceeded"
                else:
                    status = "on_track"
            else:
                status = "against"

        return {
            "src_idx": src_idx,
            "end_idx": end_idx,
            "cur_idx": cur_idx,
            "remaining": remaining,
            "current_return": current_return,
            "expected_return": expected_return,
            "direction": direction,
            "status": status,
        }

    @staticmethod
    def _format_status(prog, default_color):
        if not prog:
            return "", default_color

        cr = prog["current_return"] * 100.0
        status = prog["status"]
        rem = prog["remaining"]

        if status == "waiting":
            return "WAITING", "#8a93a6"
        if status == "on_track":
            return f"ON TRACK  {cr:+.2f}%  ({rem})", GREEN
        if status == "exceeded":
            return f"EXCEEDED  {cr:+.2f}%  ({rem})", GREEN
        if status == "against":
            return f"AGAINST  {cr:+.2f}%  ({rem})", RED
        if status == "resolved_correct":
            return f"CORRECT  {cr:+.2f}%", GREEN
        if status == "resolved_wrong":
            return f"WRONG  {cr:+.2f}%", RED
        return "", default_color

    def _draw_prediction_arrow(self, symbol, timeframe):
        if self._n == 0:
            return

        rows = self.db.get_predictions(
            symbol=symbol, timeframe=timeframe,
            limit=PREDICTIONS_HISTORY_MAX)
        if not rows:
            return

        xlim = self.ax_price.get_xlim()
        ylim = self.ax_price.get_ylim()
        rsi_ylim = (0.0, 100.0)
        active_id = (self._prediction or {}).get("prediction_id")

        dx = float(self._x[1] - self._x[0]) if self._n >= 2 else 1.0

        for p in rows:
            x_pred = _ts_to_mpl(p["candle_ts"])
            idx = int(np.searchsorted(self._x, x_pred))
            idx = max(0, min(idx, self._n - 1))
            if abs(self._x[idx] - x_pred) > 1e-6:
                continue

            x0 = float(self._x[idx])
            if x0 < xlim[0] or x0 > xlim[1]:
                continue

            close = float(self._c[idx])
            exp_ret = float(p["expected_return"])
            target = close * (1.0 + exp_ret)

            horizon_idx = self._horizon_idx(timeframe, p["horizon_seconds"])
            x1 = x0 + horizon_idx * dx

            if p["status"] == "PENDING":
                color = ("#4fc3f7" if p["direction"] == "UP"
                         else "#ffb74d" if p["direction"] == "DOWN"
                         else "#8a93a6")
            else:
                color = GREEN if p["direction_correct"] else RED

            is_active = (active_id is not None and p["id"] == active_id)

            clipped = _clip_segment(x0, close, x1, target,
                                    xlim, ylim)
            if clipped is None:
                continue
            (cx0, cy0), (cx1, cy1) = clipped

            self.ax_price.annotate(
                "",
                xy=(cx1, cy1),
                xytext=(cx0, cy0),
                arrowprops=dict(
                    arrowstyle="->", color=color,
                    linewidth=1.8 if is_active else 1.0,
                    shrinkA=0, shrinkB=0,
                    mutation_scale=16 if is_active else 11,
                    alpha=1.0 if is_active else 0.75,
                ),
                zorder=20, annotation_clip=True,
            )
            self.ax_price.plot(
                [cx0], [cy0], marker="o",
                markersize=5 if is_active else 3,
                markerfacecolor=color, markeredgecolor=color,
                alpha=1.0 if is_active else 0.75, zorder=21,
                clip_on=True,
            )

            if is_active:
                prog = self._prediction_progress()
                status_text, status_color = self._format_status(prog, color)
                label = (f"{p['direction']} "
                         f"{p['probability'] * 100:.0f}%\n"
                         f"E {exp_ret * 100:+.2f}%")
                if status_text:
                    label += f"\n{status_text}"

                self.ax_price.annotate(
                    label,
                    xy=(cx1, cy1),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom",
                    color=status_color, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="#181c24", ec=status_color, alpha=0.9),
                    zorder=22, annotation_clip=True,
                )

                if prog and prog["status"] != "waiting":
                    cur_idx = prog["cur_idx"]
                    if cur_idx != idx:
                        cx = float(self._x[cur_idx])
                        cy = float(self._c[cur_idx])
                        clip2 = _clip_segment(x0, close, cx, cy,
                                              xlim, ylim)
                        if clip2 is not None:
                            (lx0, ly0), (lx1, ly1) = clip2
                            self.ax_price.plot(
                                [lx0, lx1], [ly0, ly1],
                                color=status_color, linewidth=1.2,
                                linestyle="--", alpha=0.75, zorder=19,
                                clip_on=True,
                            )
                            self.ax_price.plot(
                                [lx1], [ly1], marker="s", markersize=7,
                                markerfacecolor=status_color,
                                markeredgecolor=status_color, zorder=21,
                                clip_on=True,
                            )

            if is_active:
                rsi_src = float(self._rsi[idx])
                if np.isnan(rsi_src):
                    continue
                rsi_delta = (10.0 if p["direction"] == "UP"
                             else -10.0 if p["direction"] == "DOWN"
                             else 0.0)
                rsi_target = max(0.0, min(100.0, rsi_src + rsi_delta))
                rsi_clip = _clip_segment(x0, rsi_src, x1, rsi_target,
                                         xlim, rsi_ylim)
                if rsi_clip is not None:
                    (rx0, ry0), (rx1, ry1) = rsi_clip
                    if rsi_delta != 0:
                        self.ax_rsi.annotate(
                            "",
                            xy=(rx1, ry1), xytext=(rx0, ry0),
                            arrowprops=dict(
                                arrowstyle="->", color=color, linewidth=1.8,
                                shrinkA=0, shrinkB=0, mutation_scale=14,
                            ),
                            zorder=20, annotation_clip=True,
                        )
                    self.ax_rsi.plot(
                        [rx0], [ry0], marker="o", markersize=5,
                        markerfacecolor=color, markeredgecolor=color,
                        zorder=21, clip_on=True,
                    )

    def _draw_chart(self, symbol, timeframe, candles, old_xlim):
        ind = compute_indicators(candles)
        self._candles = candles

        arr = np.asarray(candles, dtype=float)
        ts_ms = arr[:, 0]
        self._o = arr[:, 1]
        self._h = arr[:, 2]
        self._l = arr[:, 3]
        self._c = arr[:, 4]
        self._v = arr[:, 5]

        self._x = _ts_to_mpl(ts_ms)
        self._n = self._x.size

        self._ema_fast = self._to_np(ind["ema_fast"])
        self._ema_slow = self._to_np(ind["ema_slow"])
        self._rsi = self._to_np(ind["rsi"])

        self._hover_idx = -1

        self.ax_price.clear()
        self._style_axes()
        self.ax_price.set_autoscaley_on(False)

        self._draw_candles(self.ax_price, self._x,
                           self._o, self._h, self._l, self._c)

        self._plot_ema(self.ax_price, self._x, self._ema_fast,
                       "#4fc3f7", "EMA20")
        self._plot_ema(self.ax_price, self._x, self._ema_slow,
                       "#ffb74d", "EMA50")

        leg = self.ax_price.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_color(FG)

        quote = self._quote_currency()
        last_ts = self._x[-1]
        last_dt = (last_ts - 719163.0) * 86400.0
        self.ax_price.set_title(
            f"{pretty_symbol(symbol)} · {timeframe} · "
            f"last={self._c[-1]:.2f} {quote} · "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(last_dt))} UTC",
            color=FG, fontsize=10)

        self.ax_rsi.clear()
        self._style_axes()
        self.ax_rsi.set_autoscaley_on(False)

        mask = ~np.isnan(self._rsi)
        if mask.any():
            self.ax_rsi.plot(self._x[mask], self._rsi[mask],
                             color="#b39ddb", linewidth=1.2, zorder=4)

        if old_xlim is not None:
            self.ax_price.set_xlim(*old_xlim)
        else:
            start = max(0, self._n - DEFAULT_VISIBLE)
            self.ax_price.set_xlim(self._x[start], self._x[-1])

        self._init_hover_artists()
        self._hover_ann.xy = (self._x[-1], self._c[-1])
        self._apply_ylim()
        self._draw_prediction_arrow(symbol, timeframe)
        self.canvas.draw()
        self._capture_background()
        self._update_statusbar()

        ts = getattr(self, "_pending_highlight_ts", None)
        if ts is not None:
            self._pending_highlight_ts = None
            self._highlight_instance({"timestamp": ts})

    @staticmethod
    def _to_np(lst):
        return np.array([np.nan if v is None else v for v in lst],
                        dtype=float)

    def _plot_ema(self, ax, x, y, color, label):
        mask = ~np.isnan(y)
        if mask.any():
            ax.plot(x[mask], y[mask], color=color, linewidth=1.4,
                    label=label, zorder=4)

    def _draw_candles(self, ax, x, o, h, l, c):
        n = x.size
        if n < 2:
            return
        dx = x[1] - x[0]
        half = 0.35 * dx

        wick_segs = np.empty((n, 2, 2), dtype=float)
        wick_segs[:, 0, 0] = x
        wick_segs[:, 0, 1] = l
        wick_segs[:, 1, 0] = x
        wick_segs[:, 1, 1] = h
        ax.add_collection(LineCollection(
            wick_segs, colors="#5a6270", linewidths=0.6,
            zorder=2, antialiased=False))

        body = np.abs(c - o)
        tiny = np.maximum((h - l) * 0.001, 1e-8)
        body = np.where(body <= 0, tiny, body)
        bottom = np.minimum(o, c)
        top = bottom + body

        verts = np.empty((n, 4, 2), dtype=float)
        verts[:, 0, 0] = x - half
        verts[:, 0, 1] = bottom
        verts[:, 1, 0] = x - half
        verts[:, 1, 1] = top
        verts[:, 2, 0] = x + half
        verts[:, 2, 1] = top
        verts[:, 3, 0] = x + half
        verts[:, 3, 1] = bottom

        colors = np.where(c >= o, GREEN, RED)
        ax.add_collection(PolyCollection(
            verts, facecolors=colors, edgecolors=colors,
            linewidths=0.3, zorder=3, antialiased=False))

    def _apply_ylim(self):
        if self._x is None or self._n == 0:
            return
        x0, x1 = self.ax_price.get_xlim()
        mask = (self._x >= x0) & (self._x <= x1)
        if not mask.any():
            return
        ymin = float(self._l[mask].min())
        ymax = float(self._h[mask].max())

        p = self._prediction
        if p and p.get("symbol") == self._symbol \
                and p.get("timeframe") == self._timeframe:
            idx = p["idx"]
            if 0 <= idx < self._n:
                x_pred = float(self._x[idx])
                if x0 <= x_pred <= x1:
                    close = float(self._c[idx])
                    target = close * (1.0 + float(p["expected_return"]))
                    ymin = min(ymin, close, target)
                    ymax = max(ymax, close, target)

                    cur_idx = self._n - 1
                    cx = float(self._x[cur_idx])
                    cy = float(self._c[cur_idx])
                    if x0 <= cx <= x1:
                        ymin = min(ymin, cy)
                        ymax = max(ymax, cy)

        pad = max((ymax - ymin) * 0.05, 1e-9)
        self.ax_price.set_ylim(ymin - pad, ymax + pad)

    def _init_hover_artists(self):
        self._vline_price = self.ax_price.axvline(
            color=CROSSHAIR, linewidth=0.7, linestyle="--",
            alpha=0.0, zorder=10)
        self._hline_price = self.ax_price.axhline(
            color=CROSSHAIR, linewidth=0.7, linestyle="--",
            alpha=0.0, zorder=10)
        self._vline_rsi = self.ax_rsi.axvline(
            color=CROSSHAIR, linewidth=0.7, linestyle="--",
            alpha=0.0, zorder=10)

        self._hover_ann = self.ax_price.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset pixels",
            color=FG, fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="#181c24", ec=GRID, alpha=0.95),
            zorder=11, annotation_clip=False,
        )
        self._hover_ann.set_visible(False)

    def _capture_background(self):
        self._bg_price = self.canvas.copy_from_bbox(self.ax_price.bbox)
        self._bg_rsi = self.canvas.copy_from_bbox(self.ax_rsi.bbox)
        self._blit_ready = True

    def _on_resize(self, _event):
        if self._x is not None:
            self.canvas.draw()
            self._capture_background()

    def _place_tooltip(self, x0, y0, text):
        ann = self._hover_ann
        ann.xy = (x0, y0)
        ann.set_text(text)
        ann.set_visible(True)
        ann.set_ha("left")
        ann.set_va("bottom")
        ann.set_position((12, 12))

        renderer = self.canvas.get_renderer()
        bbox = ann.get_window_extent(renderer=renderer)
        ax_bbox = self.ax_price.bbox
        pad = 8
        dx = dy = 0.0

        if bbox.x1 > ax_bbox.x1 - pad:
            dx = (ax_bbox.x1 - pad) - bbox.x1
        if bbox.x0 + dx < ax_bbox.x0 + pad:
            dx = (ax_bbox.x0 + pad) - bbox.x0
        if bbox.y1 > ax_bbox.y1 - pad:
            dy = (ax_bbox.y1 - pad) - bbox.y1
        if bbox.y0 + dy < ax_bbox.y0 + pad:
            dy = (ax_bbox.y0 + pad) - bbox.y0

        ann.set_position((12 + dx, 12 + dy))

    def _on_motion(self, event):
        if self._x is None or not self._blit_ready or self._n == 0:
            return
        if event.inaxes is None or event.xdata is None:
            self._hide_hover()
            return

        idx = int(np.searchsorted(self._x, event.xdata))
        idx = max(0, min(idx, self._n - 1))
        if idx > 0 and abs(self._x[idx - 1] - event.xdata) < \
                abs(self._x[idx] - event.xdata):
            idx -= 1

        if idx == self._hover_idx and self._hover_visible:
            return
        self._hover_idx = idx

        o = self._o[idx]
        h = self._h[idx]
        l = self._l[idx]
        c = self._c[idx]
        v = self._v[idx]
        ema_f = self._ema_fast[idx]
        ema_s = self._ema_slow[idx]
        rsi_v = self._rsi[idx]

        ts_ms = (self._x[idx] - 719163.0) * 86400.0 * 1000.0
        dt_str = datetime.fromtimestamp(
            ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

        quote = self._quote_currency()

        def fmt(x):
            return "—" if np.isnan(x) else f"{x:.2f}"

        text = "\n".join([
            f"{dt_str} UTC",
            f"O {o:.2f}",
            f"H {h:.2f}",
            f"L {l:.2f}",
            f"C {c:.2f} {quote}",
            f"V {v:.2f}",
            f"EMA20 {fmt(ema_f)}",
            f"EMA50 {fmt(ema_s)}",
            f"RSI14 {fmt(rsi_v)}",
        ])

        x0 = self._x[idx]
        self._vline_price.set_xdata([x0, x0])
        self._vline_price.set_alpha(0.7)
        self._vline_rsi.set_xdata([x0, x0])
        self._vline_rsi.set_alpha(0.7)
        self._hline_price.set_ydata([c, c])
        self._hline_price.set_alpha(0.7)

        self._place_tooltip(x0, c, text)
        self._hover_visible = True
        self._blit_hover()

    def _blit_hover(self):
        self.canvas.restore_region(self._bg_price)
        self.ax_price.draw_artist(self._vline_price)
        self.ax_price.draw_artist(self._hline_price)
        self.ax_price.draw_artist(self._hover_ann)
        self.canvas.blit(self.ax_price.bbox)

        self.canvas.restore_region(self._bg_rsi)
        self.ax_rsi.draw_artist(self._vline_rsi)
        self.canvas.blit(self.ax_rsi.bbox)

    def _hide_hover(self):
        if not self._hover_visible or not self._blit_ready:
            return
        self._hover_ann.set_visible(False)
        self._vline_price.set_alpha(0.0)
        self._vline_rsi.set_alpha(0.0)
        self._hline_price.set_alpha(0.0)
        self._hover_visible = False
        self._hover_idx = -1
        self._blit_hover()

    def _on_leave(self, _event):
        self._hide_hover()

    def _on_scroll(self, event):
        if event.inaxes not in (self.ax_price, self.ax_rsi):
            return
        if event.button == "up":
            factor = 0.8
        elif event.button == "down":
            factor = 1.25
        else:
            return

        if event.xdata is not None:
            self._zoom_center = event.xdata
        else:
            xlim = self.ax_price.get_xlim()
            self._zoom_center = (xlim[0] + xlim[1]) / 2

        self._zoom_pending = (self._zoom_pending or 1.0) * factor

        if self._zoom_scheduled is None:
            self._zoom_scheduled = self.root.after(
                ZOOM_THROTTLE_MS, self._flush_zoom)

    def _flush_zoom(self):
        self._zoom_scheduled = None
        factor = self._zoom_pending
        center = self._zoom_center
        self._zoom_pending = None
        if factor is None or factor == 1.0 or self._x is None:
            return
        self._zoom(factor, center=center)

    def _zoom(self, factor, center=None):
        if self._x is None or self._n == 0:
            return
        xlim = self.ax_price.get_xlim()
        if center is None:
            center = (xlim[0] + xlim[1]) / 2
        full_width = self._x[-1] - self._x[0]
        min_width = full_width / 5000
        max_width = full_width * 1.05
        old_width = xlim[1] - xlim[0]
        new_width = max(min_width, min(old_width * factor, max_width))
        left = center - (center - xlim[0]) * (new_width / old_width)
        right = left + new_width
        self.ax_price.set_xlim(left, right)
        self._redraw_after_view_change()

    def _reset_zoom(self):
        if self._x is None or self._n == 0:
            return
        start = max(0, self._n - DEFAULT_VISIBLE)
        self.ax_price.set_xlim(self._x[start], self._x[-1])
        self._redraw_after_view_change()

    def _redraw_after_view_change(self):
        self._hover_visible = False
        self._hover_idx = -1
        if self._hover_ann is not None:
            self._hover_ann.set_visible(False)
        self._apply_ylim()
        self.canvas.draw()
        self._capture_background()
        self._draw_prediction_arrow(self._symbol, self._timeframe)
        self.canvas.draw()
        self._capture_background()

    def _open_ml_insights_window(self):
        if getattr(self, "_ml_win", None) is not None \
                and self._ml_win.winfo_exists():
            self._ml_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — ML Insights")
        win.geometry("900x640")
        win.configure(bg=BG)
        self._ml_win = win

        head = tk.Frame(win, bg=BG)
        head.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Label(head, text="Horizon:", bg=BG, fg=FG).pack(side=tk.LEFT)
        self._ml_horizon_var = tk.StringVar(value="1h")
        labels = [lbl for lbl, _ in ML_HORIZONS]
        cb = ttk.Combobox(head, textvariable=self._ml_horizon_var,
                          values=labels, state="readonly", width=6)
        cb.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Button(head, text="Reload",
                   command=self._refresh_ml_insights
                   ).pack(side=tk.LEFT)

        nb = ttk.Notebook(win)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                padx=8, pady=(0, 8))
        self._ml_nb = nb

        # ---- importance ----
        imp = tk.Frame(nb, bg=BG)
        nb.add(imp, text="Feature importance")
        cols = ("feature", "importance")
        tv = ttk.Treeview(imp, columns=cols,
                          show="headings", height=15)
        tv.heading("feature", text="feature")
        tv.column("feature", width=200, anchor="w")
        tv.heading("importance", text="importance")
        tv.column("importance", width=140, anchor="e")
        tv.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                padx=8, pady=8)
        self._ml_imp_tv = tv

        # ---- shap ----
        shp = tk.Frame(nb, bg=BG)
        nb.add(shp, text="SHAP")
        self._ml_shap_txt = tk.Text(
            shp, bg="#131720", fg=FG, font=("Consolas", 10),
            bd=0, padx=10, pady=10, wrap="word")
        self._ml_shap_txt.pack(side=tk.TOP, fill=tk.BOTH,
                               expand=True, padx=8, pady=8)
        self._ml_shap_txt.configure(state="disabled")

        # ---- calibration ----
        cal = tk.Frame(nb, bg=BG)
        nb.add(cal, text="Calibration")
        cols = ("bucket", "predicted", "actual", "count")
        ctv = ttk.Treeview(cal, columns=cols,
                           show="headings", height=15)
        for c in cols:
            ctv.heading(c, text=c)
            ctv.column(c, width=140, anchor="center")
        ctv.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                 padx=8, pady=8)
        self._ml_cal_tv = ctv

        nb.bind("<<NotebookTabChanged>>",
                lambda e: self._refresh_ml_insights())

        self._refresh_ml_insights()

    def _horizon_seconds_for_label(self, label):
        for lbl, sec in ML_HORIZONS:
            if lbl == label:
                return sec
        return ML_DEFAULT_HORIZON

    def _refresh_ml_insights(self):
        win = getattr(self, "_ml_win", None)
        if win is None or not win.winfo_exists():
            return
        if self._symbol is None:
            return

        nb = self._ml_nb
        tab = nb.index(nb.select())
        seconds = self._horizon_seconds_for_label(
            self._ml_horizon_var.get())

        if tab == 0:
            fut = self.engine.request_feature_importance(
                self._symbol, self._timeframe, seconds)
            self._fill_importance_async(fut)
        elif tab == 1:
            fut = self.engine.request_shap_explain(
                self._symbol, self._timeframe, seconds)
            self._fill_shap_async(fut)
        else:
            fut = self.engine.request_calibration(
                self._symbol, self._timeframe)
            self._fill_calibration_async(fut)

    def _fill_importance_async(self, fut):
        tv = self._ml_imp_tv
        for i in tv.get_children():
            tv.delete(i)

        def _check():
            try:
                res = fut.result(timeout=0)
            except Exception:
                self.root.after(300, _check)
                return
            if res is None:
                tv.insert("", "end",
                          values=("— no model —", ""))
                return
            for row in res["importance"]:
                tv.insert("", "end", values=(
                    row["feature"],
                    f"{row['importance'] * 100:.2f}%"))
        self.root.after(200, _check)

    def _fill_shap_async(self, fut):
        w = self._ml_shap_txt
        w.configure(state="normal")
        w.delete("1.0", tk.END)
        w.insert("1.0", "computing...")
        w.configure(state="disabled")

        def _check():
            try:
                res = fut.result(timeout=0)
            except Exception:
                self.root.after(500, _check)
                return
            if res is None or res.get("explain") is None:
                w.configure(state="normal")
                w.delete("1.0", tk.END)
                w.insert("1.0", "no model or shap not available")
                w.configure(state="disabled")
                return

            lines = [f"model: {res['version']}", ""]
            for cls in res["explain"]:
                lines.append(f"[{cls['label']}]")
                for item in cls["top"]:
                    lines.append(
                        f"  {item['feature']:<15} "
                        f"{item['value']:+.5f}")
                lines.append("")
            w.configure(state="normal")
            w.delete("1.0", tk.END)
            w.insert("1.0", "\n".join(lines))
            w.configure(state="disabled")
        self.root.after(500, _check)

    def _fill_calibration_async(self, fut):
        tv = self._ml_cal_tv
        for i in tv.get_children():
            tv.delete(i)

        def _check():
            try:
                rows = fut.result(timeout=0)
            except Exception:
                self.root.after(300, _check)
                return
            if not rows:
                tv.insert("", "end",
                          values=("—", "—", "—", "no resolved"))
                return
            for r in rows:
                tv.insert("", "end", values=(
                    r["bucket"],
                    f"{r['predicted'] * 100:.1f}%",
                    f"{r['actual'] * 100:.1f}%",
                    r["count"],
                ))
        self.root.after(200, _check)

    def _open_consensus_window(self):
        if getattr(self, "_cons_win", None) is not None \
                and self._cons_win.winfo_exists():
            self._cons_win.lift()
            self._refresh_consensus()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Consensus")
        win.geometry("620x420")
        win.configure(bg=BG)
        self._cons_win = win

        head = tk.Frame(win, bg=BG)
        head.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        tk.Label(head, text="Multi-timeframe consensus",
                 bg=BG, fg=FG,
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT)
        ttk.Button(head, text="Refresh",
                   command=self._refresh_consensus
                   ).pack(side=tk.RIGHT)

        self._cons_summary = tk.Label(
            win, text="—", bg=BG, fg=FG,
            font=("Consolas", 11, "bold"),
            anchor="w", justify=tk.LEFT)
        self._cons_summary.pack(side=tk.TOP, fill=tk.X,
                                padx=12, pady=(0, 6))

        cols = ("tf", "direction", "probability",
                "expected_return", "weight")
        tv = ttk.Treeview(win, columns=cols,
                          show="headings", height=10)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=110, anchor="center")
        tv.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                padx=8, pady=(0, 8))
        self._cons_tv = tv

        self._refresh_consensus()

    def _refresh_consensus(self):
        if self._symbol is None:
            return
        fut = self.engine.request_consensus(self._symbol)

        def _check():
            try:
                res = fut.result(timeout=0)
            except Exception:
                self.root.after(400, _check)
                return
            if res is None:
                return

            direction = res["direction"]
            score = res["score"]
            avg_ret = res["avg_expected_return"]
            color = GREEN if direction == "UP" else \
                RED if direction == "DOWN" else FG
            self._cons_summary.config(
                text=f"Consensus: {direction}   "
                     f"score={score:+.3f}   "
                     f"E[ret]={avg_ret * 100:+.3f}%",
                fg=color)

            tv = self._cons_tv
            for i in tv.get_children():
                tv.delete(i)
            for r in res["per_timeframe"]:
                tv.insert("", "end", values=(
                    r["timeframe"],
                    r["direction"],
                    f"{r['probability'] * 100:.1f}%",
                    f"{r['expected_return'] * 100:+.3f}%",
                    f"{r['weight']:.1f}",
                ))
        self.root.after(300, _check)

    def _open_network_window(self):
        if getattr(self, "_net_win", None) is not None \
                and self._net_win.winfo_exists():
            self._net_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Network")
        win.geometry("560x480")
        win.configure(bg=BG)
        self._net_win = win

        head = tk.Frame(win, bg=BG)
        head.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(12, 6))

        tk.Label(head, text="API status", bg=BG, fg=FG,
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

        self._net_status_lbl = tk.Label(head, text="●", bg=BG,
                                        fg="#8a93a6",
                                        font=("Consolas", 14))
        self._net_status_lbl.pack(side=tk.RIGHT)

        body = tk.Frame(win, bg=BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                  padx=12, pady=(0, 12))

        self._net_fields = {}
        labels = [
            ("connected", "Connected"),
            ("uptime", "Uptime"),
            ("requests", "Requests"),
            ("req_rate", "Req/sec"),
            ("errors", "Errors"),
            ("retries", "Retries"),
            ("retries_429", "Retries 429"),
            ("retry_wait", "Total wait"),
            ("bytes", "Downloaded"),
            ("rate", "Download rate"),
            ("last_lat", "Latency last"),
            ("avg_lat", "Latency avg"),
            ("min_lat", "Latency min"),
            ("max_lat", "Latency max"),
            ("last_err", "Last error"),
        ]
        for i, (key, lbl) in enumerate(labels):
            tk.Label(body, text=lbl + ":", bg=BG, fg="#8a93a6",
                     anchor="w", width=18
                     ).grid(row=i, column=0, sticky="w", pady=2)
            v = tk.Label(body, text="—", bg=BG, fg=FG,
                         anchor="w", font=("Consolas", 10))
            v.grid(row=i, column=1, sticky="w", pady=2)
            self._net_fields[key] = v

        btns = tk.Frame(win, bg=BG)
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(0, 12))

        ttk.Button(btns, text="Ping",
                   command=self._net_ping).pack(side=tk.LEFT)
        ttk.Button(btns, text="Reset stats",
                   command=self._net_reset).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text="Close",
                   command=self._close_network_window
                   ).pack(side=tk.RIGHT)

        self._net_tick()

    def _close_network_window(self):
        if getattr(self, "_net_win", None) is not None:
            try:
                self._net_win.destroy()
            except Exception:
                pass
            self._net_win = None

    def _net_ping(self):
        fut = self.engine.request_ping()

        def _check():
            if fut is None:
                return
            try:
                latency_ms, err = fut.result(timeout=0)
            except Exception:
                self.root.after(500, _check)
                return
            if err:
                self.statusbar.config(text=f"Ping error: {err}")
            else:
                self.statusbar.config(
                    text=f"Ping: {latency_ms:.0f} ms")
        self.root.after(200, _check)

    def _net_reset(self):
        self.engine.stats.reset()

    def _net_tick(self):
        if getattr(self, "_net_win", None) is None \
                or not self._net_win.winfo_exists():
            return

        s = self.engine.stats.snapshot()

        if s["connected"]:
            self._net_status_lbl.config(text="● Connected", fg=GREEN)
        else:
            self._net_status_lbl.config(text="● Disconnected", fg=RED)

        f = self._net_fields
        f["connected"].config(text="yes" if s["connected"] else "no")
        f["uptime"].config(text=self._fmt_duration(s["uptime_sec"]))
        f["requests"].config(text=str(s["requests"]))
        f["req_rate"].config(text=f"{s['req_rate']:.2f}")
        f["errors"].config(text=str(s["errors"]))
        f["retries"].config(text=str(s["retries"]))
        f["retries_429"].config(text=str(s["retries_429"]))
        f["retry_wait"].config(text=self._fmt_duration(s["retry_wait_sec"]))
        f["bytes"].config(text=format_bytes(s["bytes_downloaded"]))
        f["rate"].config(text=format_rate(s["rate_bps"]))
        f["last_lat"].config(text=self._fmt_ms(s["last_latency_ms"]))
        f["avg_lat"].config(text=self._fmt_ms(s["avg_latency_ms"]))
        f["min_lat"].config(text=self._fmt_ms(s["min_latency_ms"]))
        f["max_lat"].config(text=self._fmt_ms(s["max_latency_ms"]))
        err = s["last_error"]
        if err and len(err) > 60:
            err = err[:60] + "…"
        f["last_err"].config(text=err or "—",
                             fg=RED if err else FG)

        self._net_win.after(NETWORK_REFRESH_MS, self._net_tick)

    @staticmethod
    def _fmt_ms(v):
        return "—" if v is None else f"{v:.0f} ms"

    @staticmethod
    def _fmt_duration(sec):
        sec = int(sec or 0)
        h, r = divmod(sec, 3600)
        m, s = divmod(r, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _open_proxy_window(self):
        if getattr(self, "_proxy_win", None) is not None \
                and self._proxy_win.winfo_exists():
            self._proxy_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Proxy")
        win.geometry("420x300")
        win.configure(bg=BG)
        self._proxy_win = win

        state = self.engine.get_proxy_state()
        self._proxy_mode_var = tk.StringVar(value=state["mode"])
        self._proxy_host_var = tk.StringVar(value=state["host"])
        self._proxy_port_var = tk.StringVar(value=state["port"])
        self._proxy_user_var = tk.StringVar(value=state["user"])
        self._proxy_pass_var = tk.StringVar(value=state["password"])

        frm = tk.Frame(win, bg=BG)
        frm.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                 padx=12, pady=12)

        tk.Label(frm, text="Mode:", bg=BG, fg=FG, anchor="w"
                 ).grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(frm, textvariable=self._proxy_mode_var,
                     values=PROXY_MODES, state="readonly", width=14
                     ).grid(row=0, column=1, sticky="w", pady=4)

        tk.Label(frm, text="Host:", bg=BG, fg=FG, anchor="w"
                 ).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=self._proxy_host_var,
                  width=24).grid(row=1, column=1, sticky="w", pady=4)

        tk.Label(frm, text="Port:", bg=BG, fg=FG, anchor="w"
                 ).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=self._proxy_port_var,
                  width=10).grid(row=2, column=1, sticky="w", pady=4)

        tk.Label(frm, text="Username:", bg=BG, fg=FG, anchor="w"
                 ).grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=self._proxy_user_var,
                  width=24).grid(row=3, column=1, sticky="w", pady=4)

        tk.Label(frm, text="Password:", bg=BG, fg=FG, anchor="w"
                 ).grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=self._proxy_pass_var,
                  width=24, show="*"
                  ).grid(row=4, column=1, sticky="w", pady=4)

        btns = tk.Frame(win, bg=BG)
        btns.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(0, 12))
        ttk.Button(btns, text="Apply",
                   command=self._apply_proxy).pack(side=tk.LEFT)
        ttk.Button(btns, text="Cancel",
                   command=self._proxy_win.destroy).pack(side=tk.RIGHT)

    def _apply_proxy(self):
        self.engine.set_proxy(
            self._proxy_mode_var.get(),
            self._proxy_host_var.get().strip(),
            self._proxy_port_var.get().strip(),
            self._proxy_user_var.get().strip(),
            self._proxy_pass_var.get().strip(),
        )
        mode = self._proxy_mode_var.get()
        host = self._proxy_host_var.get().strip()
        port = self._proxy_port_var.get().strip()
        if mode.lower() == "direct" or not host:
            self.statusbar.config(text="Proxy: Direct")
        else:
            self.statusbar.config(text=f"Proxy: {mode} {host}:{port}")
        self._proxy_win.destroy()

    def _open_predictions_window(self):
        if getattr(self, "_preds_win", None) is not None \
                and self._preds_win.winfo_exists():
            self._preds_win.lift()
            self._refresh_predictions_window()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Predictions")
        win.geometry("1100x600")
        win.configure(bg=BG)
        self._preds_win = win

        head = tk.Frame(win, bg=BG)
        head.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self._preds_stats_lbl = tk.Label(head, text="—", bg=BG, fg=FG,
                                         font=("Consolas", 10), anchor="w")
        self._preds_stats_lbl.pack(side=tk.LEFT)

        ttk.Button(head, text="Refresh",
                   command=self._refresh_predictions_window
                   ).pack(side=tk.RIGHT)

        self._preds_filter = tk.StringVar(value="ALL")
        fbox = ttk.Combobox(head, textvariable=self._preds_filter,
                            values=["ALL", "PENDING", "RESOLVED"],
                            state="readonly", width=10)
        fbox.pack(side=tk.RIGHT, padx=(0, 8))
        fbox.bind("<<ComboboxSelected>>",
                  lambda e: self._refresh_predictions_window())

        cols = ("created", "symbol", "tf", "dir", "p",
                "exp", "actual", "status", "correct", "err")
        tv = ttk.Treeview(win, columns=cols, show="headings", height=25)
        titles = {
            "created": "Created", "symbol": "Symbol", "tf": "TF",
            "dir": "Pred", "p": "P", "exp": "E[ret]",
            "actual": "Actual", "status": "Status",
            "correct": "Hit", "err": "Err",
        }
        widths = {
            "created": 140, "symbol": 100, "tf": 50,
            "dir": 60, "p": 60, "exp": 80,
            "actual": 80, "status": 90,
            "correct": 50, "err": 80,
        }
        for c in cols:
            tv.heading(c, text=titles[c])
            tv.column(c, width=widths[c], anchor="center")
        tv.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self._preds_tv = tv

        tv.tag_configure("correct", foreground=GREEN)
        tv.tag_configure("wrong", foreground=RED)
        tv.tag_configure("pending", foreground="#8a93a6")

        self._refresh_predictions_window()

    def _open_logs_window(self):
        if getattr(self, "_logs_win", None) is not None \
                and self._logs_win.winfo_exists():
            self._logs_win.lift()
            self._refresh_logs_window()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Logs")
        win.geometry("1200x600")
        win.configure(bg=BG)
        self._logs_win = win

        head = tk.Frame(win, bg=BG)
        head.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Label(head, text="Logs", bg=BG, fg=FG,
                 font=("Consolas", 10, "bold")).pack(side=tk.LEFT)

        ttk.Button(head, text="Refresh",
                   command=self._refresh_logs_window
                   ).pack(side=tk.RIGHT)

        self._logs_tab = tk.StringVar(value="parse_runs")
        ttk.Combobox(head, textvariable=self._logs_tab,
                     values=["parse_runs", "history_requests"],
                     state="readonly", width=16
                     ).pack(side=tk.RIGHT, padx=(0, 8))

        cols_parse = ("id", "started", "finished", "source",
                      "symbol", "tf", "pages", "received",
                      "inserted", "status", "error")
        cols_hist = ("id", "requested", "source",
                     "symbol", "tf", "from", "to",
                     "pages", "received", "status", "error")

        tv = ttk.Treeview(win, columns=cols_parse,
                          show="headings", height=25)
        for c in cols_parse:
            tv.heading(c, text=c)
            tv.column(c, width=90, anchor="center")
        tv.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                padx=8, pady=(0, 8))
        tv.tag_configure("error", foreground=RED)
        self._logs_tv = tv
        self._logs_cols = {"parse_runs": cols_parse,
                           "history_requests": cols_hist}

        self._refresh_logs_window()

    def _open_simulation_window(self):
        if getattr(self, "_sim_win", None) is not None \
                and self._sim_win.winfo_exists():
            self._sim_win.lift()
            self._refresh_simulation_window()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Simulation")
        win.geometry("1200x760")
        win.configure(bg=BG)
        self._sim_win = win

        head = tk.Frame(win, bg=BG)
        head.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Label(head, text="Capital:", bg=BG, fg=FG).pack(side=tk.LEFT)
        self._sim_capital = tk.StringVar(
            value=str(SIMULATION_DEFAULT_CAPITAL))
        ttk.Entry(head, textvariable=self._sim_capital,
                  width=10).pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(head, text="SL %:", bg=BG, fg=FG).pack(side=tk.LEFT)
        self._sim_sl = tk.StringVar(
            value=str(SIMULATION_STOP_LOSS_PCT * 100.0))
        ttk.Entry(head, textvariable=self._sim_sl,
                  width=6).pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(head, text="TP %:", bg=BG, fg=FG).pack(side=tk.LEFT)
        self._sim_tp = tk.StringVar(
            value=str(SIMULATION_TAKE_PROFIT_PCT * 100.0))
        ttk.Entry(head, textvariable=self._sim_tp,
                  width=6).pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(head, text="Sizing:", bg=BG, fg=FG).pack(side=tk.LEFT)
        self._sim_sizing = tk.StringVar(value=SIMULATION_SIZING_MODE)
        ttk.Combobox(head, textvariable=self._sim_sizing,
                     values=SIMULATION_SIZING_OPTIONS,
                     state="readonly", width=10
                     ).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Button(head, text="Run",
                   command=self._run_simulation
                   ).pack(side=tk.LEFT)
        ttk.Button(head, text="Refresh",
                   command=self._refresh_simulation_window
                   ).pack(side=tk.RIGHT)

        self._sim_stats = tk.Label(win, text="—", bg=BG, fg=FG,
                                   font=("Consolas", 10), anchor="w",
                                   justify=tk.LEFT)
        self._sim_stats.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        eq_frame = tk.Frame(win, bg=BG, height=220)
        eq_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 8))
        eq_frame.pack_propagate(False)
        self._sim_fig = Figure(figsize=(12, 2.2), dpi=100, facecolor=BG)
        self._sim_ax = self._sim_fig.add_subplot(111)
        self._sim_ax.set_facecolor(BG)
        for s in self._sim_ax.spines.values():
            s.set_color(GRID)
        self._sim_ax.tick_params(colors=FG, labelsize=8)
        self._sim_ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
        self._sim_canvas = FigureCanvasTkAgg(self._sim_fig,
                                             master=eq_frame)
        self._sim_canvas.get_tk_widget().configure(
            bg=BG, highlightthickness=0)
        self._sim_canvas.get_tk_widget().pack(
            side=tk.TOP, fill=tk.BOTH, expand=True)

        cols = ("id", "created", "initial", "final",
                "profit", "ret%", "trades", "wins", "losses",
                "winrate", "avg", "dd%", "size", "sl", "tp")
        tv = ttk.Treeview(win, columns=cols, show="headings", height=12)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=80, anchor="center")
        tv.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                padx=8, pady=(0, 8))
        tv.bind("<<TreeviewSelect>>",
                lambda e: self._on_simulation_select())
        self._sim_tv = tv

        self._refresh_simulation_window()

    def _run_simulation(self):
        if self._symbol is None:
            return
        try:
            capital = float(self._sim_capital.get())
        except ValueError:
            capital = SIMULATION_DEFAULT_CAPITAL
        try:
            sl = float(self._sim_sl.get()) / 100.0
        except ValueError:
            sl = SIMULATION_STOP_LOSS_PCT
        try:
            tp = float(self._sim_tp.get()) / 100.0
        except ValueError:
            tp = SIMULATION_TAKE_PROFIT_PCT
        sizing = self._sim_sizing.get() or SIMULATION_SIZING_MODE

        self.lbl_status.config(text="Running simulation...")
        record, trades = run_simulation(
            self.db, self._symbol, self._timeframe,
            initial_capital=capital,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            sizing_mode=sizing)
        self._last_sim = (record, trades)
        self.lbl_status.config(
            text=f"Simulation: {record['trades']} trades · "
                 f"return={record['return_pct'] * 100:+.2f}%")
        self._refresh_simulation_window()
        self._show_simulation_details(record, trades)
        self._draw_equity_curve(record)

    def _refresh_simulation_window(self):
        tv = getattr(self, "_sim_tv", None)
        if tv is None:
            return
        for i in tv.get_children():
            tv.delete(i)
        rows = self.db.get_simulations(self._symbol, self._timeframe,
                                       limit=200)
        for r in rows:
            try:
                cfg = json.loads(r.get("config_json") or "{}")
            except Exception:
                cfg = {}
            sl = cfg.get("stop_loss_pct")
            tp = cfg.get("take_profit_pct")
            sizing = cfg.get("sizing_mode", "")
            tv.insert("", "end", values=(
                r["id"],
                self._fmt_ts(r["created_at"] * 1000),
                f"{r['initial_capital']:.2f}",
                f"{r['final_capital']:.2f}",
                f"{r['profit']:+.2f}",
                f"{r['return_pct'] * 100:+.2f}",
                r["trades"], r["wins"], r["losses"],
                f"{r['win_rate'] * 100:.1f}%",
                f"{r['avg_trade']:+.2f}",
                f"{r['max_drawdown'] * 100:.2f}",
                sizing,
                f"{sl * 100:.1f}%" if sl is not None else "",
                f"{tp * 100:.1f}%" if tp is not None else "",
            ))

    def _on_simulation_select(self):
        tv = getattr(self, "_sim_tv", None)
        if tv is None:
            return
        sel = tv.selection()
        if not sel:
            return
        vals = tv.item(sel[0], "values")
        if not vals:
            return
        sim_id = int(vals[0])
        trades = self.db.get_simulation_trades(sim_id)
        record = {
            "id": sim_id,
            "symbol": self._symbol,
            "timeframe": self._timeframe,
            "initial_capital": float(vals[2]),
            "final_capital": float(vals[3]),
            "profit": float(vals[4].replace("+", "")),
            "return_pct": float(vals[5].replace("+", "")) / 100.0,
            "trades": int(vals[6]),
            "wins": int(vals[7]),
            "losses": int(vals[8]),
            "win_rate": float(vals[9].replace("%", "")) / 100.0,
            "avg_trade": float(vals[10]),
            "max_drawdown": float(vals[11]) / 100.0,
        }
        self._show_simulation_details(record, trades)
        self._draw_equity_curve_from_trades(record, trades)

    def _draw_equity_curve_from_trades(self, record, trades):
        if not trades:
            self._sim_ax.clear()
            self._style_sim_ax()
            self._sim_canvas.draw_idle()
            return
        cap = record["initial_capital"]
        curve = [(trades[0]["entry_ts"], cap)]
        for t in trades:
            cap += t["pnl"]
            curve.append((t["exit_ts"], cap))
        self._sim_ax.clear()
        self._style_sim_ax()
        xs = [c[0] / 1000.0 for c in curve]
        ys = [c[1] for c in curve]
        color = GREEN if cap >= record["initial_capital"] else RED
        self._sim_ax.plot(xs, ys, color=color, linewidth=1.6)
        self._sim_ax.axhline(record["initial_capital"],
                             color="#444a5a", linestyle="--",
                             linewidth=0.7)
        self._sim_ax.set_ylabel("Equity", color=FG, fontsize=8)
        self._sim_canvas.draw_idle()

    def _draw_equity_curve(self, record):
        curve = record.get("equity_curve") or []
        if not curve:
            return
        self._sim_ax.clear()
        self._style_sim_ax()
        xs = [c[0] / 1000.0 for c in curve]
        ys = [c[1] for c in curve]
        color = GREEN if record["return_pct"] >= 0 else RED
        self._sim_ax.plot(xs, ys, color=color, linewidth=1.6)
        self._sim_ax.axhline(record["initial_capital"],
                             color="#444a5a", linestyle="--",
                             linewidth=0.7)
        self._sim_ax.set_ylabel("Equity", color=FG, fontsize=8)
        self._sim_canvas.draw_idle()

    def _style_sim_ax(self):
        self._sim_ax.set_facecolor(BG)
        for s in self._sim_ax.spines.values():
            s.set_color(GRID)
        self._sim_ax.tick_params(colors=FG, labelsize=8)
        self._sim_ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)

    def _show_simulation_details(self, record, trades):
        lines = [
            f"Simulation #{record['id']}",
            f"Symbol: {pretty_symbol(record['symbol'])} "
            f"{record['timeframe']}",
            f"Initial: {record['initial_capital']:.2f} USDT",
            f"Final:   {record['final_capital']:.2f} USDT",
            f"Profit:  {record['profit']:+.2f} USDT "
            f"({record['return_pct'] * 100:+.2f}%)",
            f"Trades:  {record['trades']} "
            f"(W {record['wins']} / L {record['losses']})",
            f"Win rate: {record['win_rate'] * 100:.1f}%",
            f"Avg trade: {record['avg_trade']:+.4f}",
            f"Max drawdown: {record['max_drawdown'] * 100:.2f}%",
        ]
        self._sim_stats.config(text="   |   ".join(lines[:4]) + "\n" +
                                    "   |   ".join(lines[4:]))

    def _open_export_window(self):
        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Export")
        win.geometry("520x360")
        win.configure(bg=BG)

        tk.Label(win, text="Table:", bg=BG, fg=FG
                 ).pack(anchor="w", padx=12, pady=(12, 2))
        self._export_table_var = tk.StringVar(value="candles")
        tables = ["candles", "features", "predictions", "models",
                  "genomes", "genome_instances", "parse_runs",
                  "history_requests", "simulations",
                  "simulation_trades"]
        ttk.Combobox(win, textvariable=self._export_table_var,
                     values=tables, state="readonly", width=30
                     ).pack(anchor="w", padx=12)

        tk.Label(win, text="Format:", bg=BG, fg=FG
                 ).pack(anchor="w", padx=12, pady=(12, 2))
        self._export_fmt_var = tk.StringVar(value="xlsx")
        ttk.Combobox(win, textvariable=self._export_fmt_var,
                     values=EXPORT_FORMATS, state="readonly", width=30
                     ).pack(anchor="w", padx=12)

        tk.Label(win, text="Limit rows:", bg=BG, fg=FG
                 ).pack(anchor="w", padx=12, pady=(12, 2))
        self._export_limit_var = tk.StringVar(value="100000")
        ttk.Entry(win, textvariable=self._export_limit_var,
                  width=12).pack(anchor="w", padx=12)

        ttk.Button(win, text="Export...",
                   command=self._do_export
                   ).pack(anchor="w", padx=12, pady=(16, 4))

    def _do_export(self):
        table = self._export_table_var.get()
        fmt = self._export_fmt_var.get()
        try:
            limit = int(self._export_limit_var.get())
        except ValueError:
            limit = 100000

        initial = os.path.join(EXPORT_DEFAULT_DIR,
                               f"{table}.{fmt}")
        os.makedirs(EXPORT_DEFAULT_DIR, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save export",
            initialfile=os.path.basename(initial),
            initialdir=EXPORT_DEFAULT_DIR,
            defaultextension=f".{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}"),
                       ("All files", "*.*")])
        if not path:
            return

        rows, columns = self._fetch_export(table, limit)
        export_rows(rows, columns, path, fmt)
        self.statusbar.config(text=f"Exported {len(rows)} rows -> {path}")

    def _fetch_export(self, table, limit):
        if table == "candles":
            rows = self.db.get_candles_with_ids(self._symbol, self._timeframe)
            rows = rows[:limit]
            data = [{"id": r[0], "timestamp": r[1], "open": r[2],
                     "high": r[3], "low": r[4], "close": r[5],
                     "volume": r[6]} for r in rows]
            cols = ["id", "timestamp", "open", "high", "low",
                    "close", "volume"]
            return data, cols
        if table == "features":
            raw = self.db.get_features(self._symbol, self._timeframe,
                                       FEATURE_SCHEMA_VERSION)
            data = []
            for r in raw[:limit]:
                data.append({
                    "candle_id": r[0],
                    "return_1": r[1], "return_5": r[2], "return_20": r[3],
                    "rsi": r[4], "ema20": r[5], "ema50": r[6],
                    "volatility": r[7], "volume_ratio": r[8],
                    "high_low_range": r[9], "body_size": r[10],
                    "upper_wick": r[11], "lower_wick": r[12],
                })
            cols = ["candle_id", "return_1", "return_5", "return_20",
                    "rsi", "ema20", "ema50", "volatility",
                    "volume_ratio", "high_low_range", "body_size",
                    "upper_wick", "lower_wick"]
            return data, cols
        if table == "predictions":
            return self.db.get_predictions(self._symbol, self._timeframe,
                                           limit=limit), \
                ["id", "candle_id", "symbol", "timeframe", "candle_ts",
                 "direction", "probability", "expected_return",
                 "horizon_seconds", "flat_threshold",
                 "model_version", "created_at",
                 "actual_direction", "actual_return",
                 "direction_correct", "return_error", "accuracy",
                 "status", "resolved_at"]
        if table == "models":
            rows = self.db.get_latest_model(self._symbol, self._timeframe)
            return ([rows] if rows else []), \
                ["version", "symbol", "timeframe", "horizon_seconds",
                 "flat_threshold", "feature_schema_version",
                 "n_train", "cls_accuracy", "reg_mae",
                 "trained_at", "cls_path", "reg_path"]
        if table == "genomes":
            return self.db.get_genomes(self._symbol), \
                ["id", "genome_uid", "symbol", "name",
                 "genesis_instance_id", "created_at",
                 "feature_schema_version"]
        if table == "genome_instances":
            gens = self.db.get_genomes(self._symbol)
            all_rows = []
            for g in gens:
                all_rows.extend(self.db.get_genome_instances(g["id"]))
            return all_rows[:limit], \
                ["id", "genome_id", "instance_number", "timestamp",
                 "symbol", "timeframe", "candle_id", "fingerprint",
                 "similarity_to_genesis", "similarity_to_consensus",
                 "feature_vector"]
        if table == "parse_runs":
            return self.db.get_parse_runs(self._symbol, limit), \
                ["id", "started_at", "finished_at", "source",
                 "symbol", "timeframe", "pages",
                 "candles_received", "candles_inserted",
                 "status", "error"]
        if table == "history_requests":
            return self.db.get_history_requests(self._symbol, limit), \
                ["id", "requested_at", "source", "symbol", "timeframe",
                 "from_ts", "to_ts", "pages", "candles_received",
                 "status", "error"]
        if table == "simulations":
            return self.db.get_simulations(self._symbol,
                                           self._timeframe, limit), \
                ["id", "created_at", "symbol", "timeframe",
                 "initial_capital", "final_capital", "profit", "return_pct",
                 "trades", "wins", "losses", "win_rate", "avg_trade",
                 "max_drawdown", "config_json"]
        if table == "simulation_trades":
            sims = self.db.get_simulations(self._symbol,
                                           self._timeframe, limit=5)
            all_t = []
            for s in sims:
                all_t.extend(self.db.get_simulation_trades(s["id"]))
            return all_t[:limit], \
                ["id", "simulation_id", "prediction_id",
                 "entry_ts", "exit_ts", "entry_price", "exit_price",
                 "direction", "size", "pnl", "return_pct"]
        return [], []

    def _refresh_logs_window(self):
        tv = getattr(self, "_logs_tv", None)
        if tv is None:
            return

        for i in tv.get_children():
            tv.delete(i)

        which = self._logs_tab.get()
        if which == "parse_runs":
            rows = self.db.get_parse_runs(self._symbol, limit=300)
        else:
            rows = self.db.get_history_requests(self._symbol, limit=300)

        cols = self._logs_cols[which]
        tv.configure(columns=cols)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=90, anchor="center")

        for r in rows:
            if which == "parse_runs":
                vals = (
                    r["id"],
                    self._fmt_ts(r["started_at"] * 1000),
                    self._fmt_ts(r["finished_at"] * 1000)
                    if r["finished_at"] else "",
                    r["source"], pretty_symbol(r["symbol"]),
                    r["timeframe"], r["pages"] or "",
                    r["candles_received"] or "",
                    r["candles_inserted"] or "",
                    r["status"], r["error"] or "",
                )
                tag = "error" if r["status"] != "OK" else ""
            else:
                vals = (
                    r["id"],
                    self._fmt_ts(r["requested_at"] * 1000),
                    r["source"], pretty_symbol(r["symbol"]),
                    r["timeframe"],
                    self._fmt_ts(r["from_ts"]),
                    self._fmt_ts(r["to_ts"]),
                    r["pages"] or "", r["candles_received"] or "",
                    r["status"], r["error"] or "",
                )
                tag = "error" if r["status"] != "OK" else ""
            tv.insert("", "end", values=vals, tags=(tag,))

    def _refresh_predictions_window(self):
        if not getattr(self, "_preds_tv", None):
            return

        tv = self._preds_tv
        for i in tv.get_children():
            tv.delete(i)

        status = None if self._preds_filter.get() == "ALL" \
            else self._preds_filter.get()
        rows = self.db.get_predictions(
            symbol=self._symbol, timeframe=self._timeframe,
            status=status, limit=500)

        for p in rows:
            created = datetime.fromtimestamp(
                p["created_at"], tz=timezone.utc).strftime("%m-%d %H:%M:%S")

            hit = {1: "✓", 0: "✗"}.get(p["direction_correct"], "")
            actual = (f"{p['actual_direction']} "
                      f"{p['actual_return'] * 100:+.2f}%"
                      if p["actual_direction"] else "")
            err = (f"{p['return_error'] * 100:.3f}%"
                   if p["return_error"] is not None else "")

            if p["status"] == "PENDING":
                tag = "pending"
            else:
                tag = "correct" if p["direction_correct"] else "wrong"

            tv.insert("", "end", values=(
                created,
                pretty_symbol(p["symbol"]),
                p["timeframe"],
                p["direction"],
                f"{p['probability'] * 100:.1f}%",
                f"{p['expected_return'] * 100:+.2f}%",
                actual,
                p["status"],
                hit,
                err,
            ), tags=(tag,))

        st = self.db.prediction_stats(self._symbol, self._timeframe)
        if st and st["total"]:
            hits = st["hits"] or 0
            res = st["resolved"] or 0
            acc = (hits / res * 100) if res else 0
            mae = (st["mae"] or 0) * 100
            self._preds_stats_lbl.config(
                text=(f"{pretty_symbol(self._symbol or '')} "
                      f"{self._timeframe or ''} · "
                      f"total={st['total']} "
                      f"pending={st['pending']} "
                      f"resolved={res} · "
                      f"hit-rate={acc:.1f}% · "
                      f"MAE={mae:.3f}%")
            )
        else:
            self._preds_stats_lbl.config(text="No predictions")

    def _open_genomes_window(self):
        if getattr(self, "_genomes_win", None) is not None \
                and self._genomes_win.winfo_exists():
            self._genomes_win.lift()
            self._refresh_genomes_window()
            return

        win = tk.Toplevel(self.root)
        win.title("CCSMonitor — Genomes")
        win.geometry("1200x700")
        win.configure(bg=BG)
        self._genomes_win = win

        head = tk.Frame(win, bg=BG)
        head.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Label(head, text="Genome tree", bg=BG, fg=FG,
                 font=("Consolas", 10, "bold")).pack(side=tk.LEFT)

        ttk.Button(head, text="Refresh",
                   command=self._refresh_genomes_window
                   ).pack(side=tk.RIGHT)

        body = tk.Frame(win, bg=BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                  padx=8, pady=(0, 8))

        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tv = ttk.Treeview(left, show="tree", height=30)
        tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(left, orient="vertical", command=tv.yview)
        sb.pack(side=tk.RIGHT, fill="y")
        tv.configure(yscrollcommand=sb.set)

        tv.bind("<<TreeviewSelect>>", self._on_genome_tree_select)
        tv.bind("<Double-1>", self._on_genome_tree_double_click)
        self._genomes_tv = tv

        right = tk.Frame(body, bg="#131720", width=420)
        right.pack(side=tk.RIGHT, fill="y", padx=(8, 0))
        right.pack_propagate(False)

        self._genome_inspector = tk.Text(
            right, bg="#131720", fg=FG, font=("Consolas", 10),
            bd=0, padx=10, pady=10, wrap="word")
        self._genome_inspector.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._genome_inspector.configure(state="disabled")

        tv.tag_configure("coin", foreground="#ffb74d")
        tv.tag_configure("genome", foreground="#4fc3f7")
        tv.tag_configure("genesis", foreground=GREEN)
        tv.tag_configure("instance", foreground=FG)

        self._refresh_genomes_window()

    def _refresh_genomes_window(self):
        tv = getattr(self, "_genomes_tv", None)
        if tv is None:
            return

        for i in tv.get_children():
            tv.delete(i)

        root_node = tv.insert("", "end", text="CCSMonitor", open=True,
                              tags=("root",))
        self._genome_tree_map = {}

        symbols = [self._symbol] if self._symbol else SYMBOLS
        for sym in symbols:
            genomes = self.db.get_genomes(sym)
            if not genomes:
                continue
            coin_node = tv.insert(root_node, "end",
                                  text=pretty_symbol(sym),
                                  open=False, tags=("coin",))
            self._genome_tree_map[coin_node] = ("coin", sym)

            for g in genomes:
                label = f"{g['name']}   ({g['genome_uid']})"
                g_node = tv.insert(coin_node, "end", text=label,
                                   open=False, tags=("genome",))
                self._genome_tree_map[g_node] = ("genome", g["id"])

                instances = self.db.get_genome_instances(g["id"])
                for inst in instances:
                    n = inst["instance_number"]
                    if n == 0:
                        tag = "genesis"
                        txt = (f"Genesis #{n}  ·  "
                               f"{self._fmt_ts(inst['timestamp'])}")
                    else:
                        tag = "instance"
                        txt = (f"#{n}  ·  sim="
                               f"{(inst['similarity_to_genesis'] or 0) * 100:.2f}%"
                               f"  ·  {self._fmt_ts(inst['timestamp'])}")
                    i_node = tv.insert(g_node, "end", text=txt, tags=(tag,))
                    self._genome_tree_map[i_node] = ("instance", inst["id"])

    @staticmethod
    def _fmt_ts(ts_ms):
        return datetime.fromtimestamp(
            ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    def _on_genome_tree_select(self, _event):
        tv = self._genomes_tv
        sel = tv.selection()
        if not sel:
            return
        kind, ident = self._genome_tree_map.get(sel[0], (None, None))
        if kind is None:
            return

        if kind == "coin":
            self._show_genome_inspector_coin(ident)
        elif kind == "genome":
            self._show_genome_inspector_genome(ident)
        elif kind == "instance":
            self._show_genome_inspector_instance(ident)

    def _on_genome_tree_double_click(self, _event):
        tv = self._genomes_tv
        sel = tv.selection()
        if not sel:
            return
        kind, ident = self._genome_tree_map.get(sel[0], (None, None))
        if kind == "instance":
            self._goto_genome_instance(ident)

    def _goto_genome_instance(self, instance_id):
        inst = self.db.get_genome_instance_by_id(instance_id)
        if not inst:
            return
        genome = self.db.get_genome_by_id(inst["genome_id"])
        if not genome:
            return

        if (inst["symbol"] != self._symbol
                or inst["timeframe"] != self._timeframe
                or self._x is None):
            self.symbol_var.set(pretty_symbol(inst["symbol"]))
            self.tf_var.set(inst["timeframe"])
            self.on_load()
            self._pending_highlight_ts = inst["timestamp"]
            return

        self._highlight_instance(inst)

    def _highlight_instance(self, inst):
        if self._x is None or self._n == 0:
            return

        x = _ts_to_mpl(inst["timestamp"])
        if x < self._x[0] or x > self._x[-1]:
            return

        half = self._timeframe_half_width()
        self.ax_price.set_xlim(x - half, x + half)
        self.ax_rsi.set_xlim(x - half, x + half)

        self.ax_price.axvspan(x - half * 0.1, x + half * 0.1,
                              color="#4fc3f7", alpha=0.15, zorder=1)
        self._apply_ylim()
        self.canvas.draw()
        self._capture_background()

    def _timeframe_half_width(self):
        if self._n < 2:
            return 1.0
        dx = float(self._x[1] - self._x[0])
        return dx * 30

    def _show_genome_inspector_coin(self, symbol):
        genomes = self.db.get_genomes(symbol)
        total_inst = self.db.count_genome_instances(symbol, self._timeframe)
        lines = [
            "COIN",
            "",
            f"Symbol: {pretty_symbol(symbol)}",
            f"Genomes: {len(genomes)}",
            f"Instances ({self._timeframe}): {total_inst}",
        ]
        self._set_inspector_text("\n".join(lines))

    def _show_genome_inspector_genome(self, genome_id):
        g = self.db.get_genome_by_id(genome_id)
        if not g:
            return
        stats = self.db.get_genome_stats(genome_id)
        avg = stats["avg_sim"]
        avg_s = f"{avg * 100:.2f}%" if avg is not None else "—"
        lines = [
            "GENOME",
            "",
            f"ID:           {g['id']}",
            f"UID:          {g['genome_uid']}",
            f"Name:         {g['name']}",
            f"Symbol:       {pretty_symbol(g['symbol'])}",
            f"Instances:    {stats['total']}",
            f"Avg sim:      {avg_s}",
            f"Feature sch:  {g['feature_schema_version']}",
            f"Created:      {self._fmt_ts(g['created_at'] * 1000)}",
        ]
        self._set_inspector_text("\n".join(lines))

    def _show_genome_inspector_instance(self, instance_id):
        inst = self.db.get_genome_instance_by_id(instance_id)
        if not inst:
            return
        g = self.db.get_genome_by_id(inst["genome_id"])
        sim = inst["similarity_to_genesis"]
        sim_s = f"{sim * 100:.2f}%" if sim is not None else "—"
        role = "Genesis" if inst["instance_number"] == 0 else "Instance"

        lines = [
            f"{role.upper()} #{inst['instance_number']}",
            "",
            f"Genome:       {g['name'] if g else '—'}",
            f"Symbol:       {pretty_symbol(inst['symbol'])}",
            f"Timeframe:    {inst['timeframe']}",
            f"Timestamp:    {self._fmt_ts(inst['timestamp'])} UTC",
            f"Similarity:   {sim_s}",
            f"Fingerprint:  {inst['fingerprint'][:16]}…",
        ]

        if inst["feature_vector"]:
            vals = inst["feature_vector"].split(",")
            lines += ["", "Feature vector (normalized):"]
            for i, name in enumerate(FEATURE_COLS):
                if i < len(vals):
                    lines.append(f"  {name:<15} {float(vals[i]):+.4f}")

        self._set_inspector_text("\n".join(lines))

    def _set_inspector_text(self, text):
        w = getattr(self, "_genome_inspector", None)
        if w is None:
            return
        w.configure(state="normal")
        w.delete("1.0", tk.END)
        w.insert("1.0", text)
        w.configure(state="disabled")