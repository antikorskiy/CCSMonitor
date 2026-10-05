# tray.py
import threading

from PIL import Image, ImageDraw

try:
    import pystray
    HAS_TRAY = True
except Exception:
    HAS_TRAY = False


def _make_icon(color="#4fc3f7"):
    img = Image.new("RGBA", (64, 64), (14, 16, 22, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([6, 30, 14, 50], fill="#00c878")
    d.line([(10, 40), (10, 20)], fill="#00c878", width=3)
    d.rectangle([20, 22, 28, 50], fill="#dc3c50")
    d.line([(24, 40), (24, 14)], fill="#dc3c50", width=3)
    d.rectangle([34, 10, 42, 50], fill="#00c878")
    d.line([(38, 30), (38, 6)], fill="#00c878", width=3)
    d.rectangle([48, 26, 56, 50], fill="#dc3c50")
    d.line([(52, 40), (52, 18)], fill="#dc3c50", width=3)
    d.line([(6, 52), (58, 52)], fill=color, width=2)
    return img


class TrayIcon:
    def __init__(self, on_show, on_quit,
                 on_train=None, on_predict=None,
                 on_sync=None, on_toggle_alerts=None):
        self.on_show = on_show
        self.on_quit = on_quit
        self.on_train = on_train
        self.on_predict = on_predict
        self.on_sync = on_sync
        self.on_toggle_alerts = on_toggle_alerts
        self._icon = None
        self._thread = None
        self._alerts_on = True

    def available(self):
        return HAS_TRAY

    def start(self):
        if not HAS_TRAY:
            return
        icon = pystray.Icon(
            "ccsmonitor",
            _make_icon(),
            "CCSMonitor",
            menu=pystray.Menu(
                pystray.MenuItem("Show", self._wrapped(self._show),
                                 default=True),
                pystray.MenuItem("Sync now",
                                 self._wrapped(self._sync)),
                pystray.MenuItem("Train ML",
                                 self._wrapped(self._train)),
                pystray.MenuItem("Predict",
                                 self._wrapped(self._predict)),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Toggle alerts",
                                 self._wrapped(self._toggle_alerts)),
                pystray.MenuItem("Quit",
                                 self._wrapped(self._quit)),
            ),
        )
        self._icon = icon
        self._thread = threading.Thread(
            target=icon.run, daemon=True, name="TrayIcon")
        self._thread.start()

    def stop(self):
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass

    def notify(self, title, body):
        if self._icon is None:
            raise RuntimeError("tray not running")
        self._icon.notify(body, title)

    def _wrapped(self, fn):
        def _cb(_icon=None, _item=None):
            try:
                if fn:
                    fn()
            except Exception:
                pass
        return _cb

    def _show(self):
        if self.on_show:
            self.on_show()

    def _quit(self):
        if self.on_quit:
            self.on_quit()

    def _sync(self):
        if self.on_sync:
            self.on_sync()

    def _train(self):
        if self.on_train:
            self.on_train()

    def _predict(self):
        if self.on_predict:
            self.on_predict()

    def _toggle_alerts(self):
        self._alerts_on = not self._alerts_on
        if self.on_toggle_alerts:
            self.on_toggle_alerts(self._alerts_on)