# market.py
import asyncio
import json
import time

import aiohttp

from config import (
    NETWORK_REQUEST_TIMEOUT_SEC,
    NETWORK_MAX_RETRIES,
    NETWORK_INITIAL_BACKOFF_SEC,
    NETWORK_MAX_BACKOFF_SEC,
    NETWORK_429_FALLBACK_WAIT_SEC,
)


class AsyncBinanceClient:
    def __init__(self, base="https://api.binance.com", proxy=None, stats=None):
        self.base = base.rstrip("/")
        self.proxy = proxy
        self.stats = stats

    def set_proxy(self, proxy):
        self.proxy = proxy

    async def _request(self, session, url, params,
                       timeout_sec=NETWORK_REQUEST_TIMEOUT_SEC,
                       max_retries=NETWORK_MAX_RETRIES):
        backoff = NETWORK_INITIAL_BACKOFF_SEC
        last_exc = None
        for _ in range(max_retries):
            try:
                t0 = time.monotonic()
                timeout = aiohttp.ClientTimeout(total=timeout_sec)
                async with session.get(url, params=params,
                                       proxy=self.proxy,
                                       timeout=timeout) as r:
                    if r.status == 429:
                        wait = self._retry_after(r) \
                            or NETWORK_429_FALLBACK_WAIT_SEC
                        if self.stats:
                            self.stats.record_429(wait)
                        await asyncio.sleep(wait)
                        continue

                    if r.status >= 500:
                        if self.stats:
                            self.stats.record_retry(backoff)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2.0,
                                      NETWORK_MAX_BACKOFF_SEC)
                        continue

                    body = await r.read()
                    r.raise_for_status()
                    latency_ms = (time.monotonic() - t0) * 1000.0
                    if self.stats:
                        self.stats.record_request(latency_ms, len(body))
                    return json.loads(body)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if self.stats:
                    self.stats.record_error(str(e))
                    self.stats.record_retry(backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, NETWORK_MAX_BACKOFF_SEC)

        raise RuntimeError(
            f"request failed after {max_retries} retries: {last_exc}")

    @staticmethod
    def _retry_after(response):
        h = response.headers.get("Retry-After")
        if not h:
            return None
        try:
            return float(h)
        except ValueError:
            return None

    async def fetch_klines(self, session, symbol, interval, limit=1000,
                           start_time=None, end_time=None):
        params = {"symbol": symbol, "interval": interval,
                  "limit": min(limit, 1000)}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        url = f"{self.base}/api/v3/klines"
        raw = await self._request(session, url, params)

        return [
            (int(k[0]), float(k[1]), float(k[2]),
             float(k[3]), float(k[4]), float(k[5]))
            for k in raw
        ]

    async def fetch_history(self, session, symbol, interval,
                            limit=1000, end_time=None):
        return await self.fetch_klines(session, symbol, interval,
                                       limit=limit, end_time=end_time)

    async def fetch_range(self, session, symbol, interval,
                          start_ms, end_ms, progress_cb=None,
                          sleep_sec=0.15):
        out = []
        cursor = start_ms
        page = 0
        while cursor < end_ms:
            batch = await self.fetch_klines(
                session, symbol, interval, limit=1000,
                start_time=cursor, end_time=end_ms)
            if not batch:
                break
            out.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            page += 1
            if progress_cb is not None:
                try:
                    progress_cb(page, len(out), last_ts)
                except Exception:
                    pass
            if cursor < end_ms and sleep_sec > 0:
                await asyncio.sleep(sleep_sec)
        return out

    async def ping(self, session):
        url = f"{self.base}/api/v3/ping"
        t0 = time.monotonic()
        try:
            await self._request(session, url, {})
            latency_ms = (time.monotonic() - t0) * 1000.0
            return latency_ms, None
        except Exception as e:
            return None, str(e)