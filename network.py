# network.py
import threading
import time
from collections import deque

from config import NETWORK_LATENCY_WINDOW


class NetworkStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.requests = 0
            self.errors = 0
            self.retries = 0
            self.retries_429 = 0
            self.total_retry_wait_sec = 0.0
            self.bytes_downloaded = 0
            self.connected = False
            self.last_error = None
            self.last_error_ts = None
            self.last_success_ts = None
            self.last_latency_ms = None
            self.latencies = deque(maxlen=NETWORK_LATENCY_WINDOW)
            self.start_ts = time.time()

    def record_request(self, latency_ms, bytes_size):
        with self._lock:
            self.requests += 1
            self.bytes_downloaded += int(bytes_size)
            self.last_latency_ms = float(latency_ms)
            self.latencies.append(float(latency_ms))
            self.connected = True
            self.last_success_ts = time.time()

    def record_error(self, message):
        with self._lock:
            self.errors += 1
            self.last_error = str(message)
            self.last_error_ts = time.time()
            self.connected = False

    def record_retry(self, wait_sec=0.0):
        with self._lock:
            self.retries += 1
            self.total_retry_wait_sec += float(wait_sec)

    def record_429(self, wait_sec):
        with self._lock:
            self.retries_429 += 1
            self.total_retry_wait_sec += float(wait_sec)

    def set_connected(self, flag):
        with self._lock:
            self.connected = bool(flag)

    def snapshot(self):
        with self._lock:
            lats = list(self.latencies)
            if lats:
                last = self.last_latency_ms
                avg = sum(lats) / len(lats)
                mn = min(lats)
                mx = max(lats)
            else:
                last = avg = mn = mx = None

            elapsed = max(1.0, time.time() - self.start_ts)
            rate_bps = self.bytes_downloaded / elapsed
            req_rate = self.requests / elapsed

            return {
                "connected": self.connected,
                "requests": self.requests,
                "errors": self.errors,
                "retries": self.retries,
                "retries_429": self.retries_429,
                "retry_wait_sec": self.total_retry_wait_sec,
                "bytes_downloaded": self.bytes_downloaded,
                "rate_bps": rate_bps,
                "req_rate": req_rate,
                "last_latency_ms": last,
                "avg_latency_ms": avg,
                "min_latency_ms": mn,
                "max_latency_ms": mx,
                "last_error": self.last_error,
                "last_error_ts": self.last_error_ts,
                "last_success_ts": self.last_success_ts,
                "uptime_sec": elapsed,
            }


def format_bytes(n):
    if n is None:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def format_rate(bps):
    if bps is None:
        return "—"
    return format_bytes(bps) + "/s"