# async_engine.py
import asyncio
import threading
import time
from typing import Callable

import aiohttp
import numpy as np

from config import (
    BINANCE_BASE, PROXY, FETCH_LIMIT, FEATURE_SCHEMA_VERSION,
    AUTO_ML_DEFAULT_ENABLED,
    AUTO_PREDICT_ON_NEW_CANDLE, AUTO_PREDICT_MIN_INTERVAL_SEC,
    AUTO_TRAIN_MIN_INTERVAL_SEC, AUTO_TRAIN_MIN_NEW_CANDLES,
    PREDICTION_RESOLVER_INTERVAL_SEC,
    GENOME_MIN_SAMPLES, GENOME_REBUILD_EVERY_NEW_CANDLES,
    HISTORY_START_MS, HISTORY_PAGE_SLEEP_SEC,
    HISTORY_MAX_PAGES_PER_CYCLE,
    PROXY_DEFAULT_MODE,
    ML_HORIZONS, ML_DEFAULT_HORIZON,
    CONSENSUS_MIN_CONFIDENCE, CONSENSUS_TIMEFRAMES,
    CONSENSUS_WEIGHTS, SHAP_BACKGROUND_SIZE, SHAP_MAX_ROWS,
)
from network import NetworkStats
from genome import (
    normalize_matrix, normalize_vector,
    fingerprint, make_genome_name, build_genomes,
)
from ml import FEATURE_COLS
from market import AsyncBinanceClient
from features import compute_features


class DataEngine:
    def __init__(self, db):
        self.db = db
        self._loop = None
        self._thread = None
        self._session = None
        self._ready = threading.Event()
        self._sync_task = None
        self._subscribers = []
        self.stats = NetworkStats()
        self._client = AsyncBinanceClient(
            base=BINANCE_BASE, proxy=PROXY, stats=self.stats)
        self._auto_ml_enabled = AUTO_ML_DEFAULT_ENABLED
        self._train_running = False
        self._predict_running = False
        self._resolver_task = None
        self._genome_running = False
        self._history_years = None
        self._proxy_mode = PROXY_DEFAULT_MODE
        self._proxy_host = ""
        self._proxy_port = ""
        self._proxy_user = ""
        self._proxy_pass = ""

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="DataEngineLoop")
        self._thread.start()
        self._ready.wait(timeout=10)

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._bootstrap())
        self._ready.set()
        self._loop.run_forever()

    async def _bootstrap(self):
        self._session = aiohttp.ClientSession()
        self._resolver_task = asyncio.create_task(self._resolver_loop())

    def stop(self):
        if self._loop is None:
            return

        async def _shutdown():
            await self.stop_auto_sync()
            if self._resolver_task is not None:
                self._resolver_task.cancel()
                try:
                    await self._resolver_task
                except asyncio.CancelledError:
                    pass
            if self._session is not None:
                await self._session.close()

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            fut.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    def subscribe(self, callback):
        self._subscribers.append(callback)

    def set_history_years(self, value):
        self._history_years = value

    @staticmethod
    def _years_to_ms(years_str):
        if not years_str or years_str == "Full":
            return None
        try:
            n = int(years_str.rstrip("y"))
        except ValueError:
            return None
        return n * 365 * 24 * 3600 * 1000

    def _emit(self, event, **kwargs):
        for cb in list(self._subscribers):
            try:
                cb(event, kwargs)
            except Exception as e:
                print(f"[engine] subscriber error: {e!r}")

    async def _resolver_loop(self):
        while True:
            try:
                await self._resolve_pending()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._emit("sync_error", error=f"resolver: {e}")
            await asyncio.sleep(PREDICTION_RESOLVER_INTERVAL_SEC)

    async def _resolve_pending(self):
        from ml import _timeframe_to_ms
        pend = await asyncio.to_thread(self.db.get_pending_predictions)
        now = int(time.time())

        for p in pend:
            if now < p["created_at"] + p["horizon_seconds"]:
                continue

            tf_ms = _timeframe_to_ms(p["timeframe"])
            horizon_idx = max(1, int(round(
                p["horizon_seconds"] * 1000 / tf_ms)))
            target_ts = p["candle_ts"] + horizon_idx * tf_ms

            src = await asyncio.to_thread(
                self.db.get_candle_close_at, p["symbol"], p["timeframe"],
                p["candle_ts"])
            dst = await asyncio.to_thread(
                self.db.get_candle_close_at, p["symbol"], p["timeframe"],
                target_ts)

            if src is None or dst is None or src <= 0:
                continue

            actual_ret = dst / src - 1.0
            thr = float(p["flat_threshold"])

            if actual_ret > thr:
                actual_dir = "UP"
            elif actual_ret < -thr:
                actual_dir = "DOWN"
            else:
                actual_dir = "FLAT"

            dir_correct = int(actual_dir == p["direction"])
            ret_err = abs(actual_ret - float(p["expected_return"]))

            prob = await asyncio.to_thread(
                self.db.get_prediction_probability, p["id"])
            prob = float(prob) if prob is not None else 0.5
            acc = prob if dir_correct else (1.0 - prob)

            await asyncio.to_thread(
                self.db.resolve_prediction, p["id"],
                actual_dir, actual_ret, dir_correct, ret_err, acc, now)

            self._emit("prediction_resolved",
                       prediction_id=p["id"],
                       symbol=p["symbol"],
                       timeframe=p["timeframe"],
                       actual_direction=actual_dir,
                       actual_return=actual_ret,
                       direction_correct=dir_correct)

    async def sync_once(self, symbol, timeframe, limit=FETCH_LIMIT):
        self._emit("sync_started", symbol=symbol, timeframe=timeframe)

        last_ts = await asyncio.to_thread(
            self.db.last_timestamp, symbol, timeframe)

        if last_ts is None:
            rows = await self._initial_load(symbol, timeframe)
        else:
            run_id = await asyncio.to_thread(
                self.db.start_parse_run, symbol, timeframe,
                "binance-rest-tail")
            try:
                rows = await self._client.fetch_klines(
                    self._session, symbol, timeframe,
                    limit=limit, start_time=last_ts)
                status, error = "OK", None
            except Exception as e:
                rows = []
                status, error = "ERROR", str(e)
            await asyncio.to_thread(
                self.db.finish_parse_run, run_id,
                1, len(rows), 0, status, error)

        inserted = 0
        if rows:
            inserted = await asyncio.to_thread(
                self.db.insert_candles, symbol, timeframe, rows)

        features_added = await asyncio.to_thread(
            self._update_features, symbol, timeframe)

        genomes_added, instances_added = await asyncio.to_thread(
            self._update_genomes, symbol, timeframe)

        candles = await asyncio.to_thread(
            self.db.get_candles, symbol, timeframe)

        self._emit("candles_updated",
                   symbol=symbol, timeframe=timeframe,
                   inserted=inserted,
                   features_added=features_added,
                   genomes_added=genomes_added,
                   instances_added=instances_added,
                   candles=candles)

        if inserted > 0 and self._auto_ml_enabled:
            asyncio.create_task(self._maybe_auto_predict(symbol, timeframe))
            asyncio.create_task(self._maybe_auto_train(symbol, timeframe))

        return inserted, candles

    async def _initial_load(self, symbol, timeframe):
        span_ms = self._years_to_ms(self._history_years)
        if span_ms is None:
            start_ms = HISTORY_START_MS
        else:
            now_ms = int(time.time() * 1000)
            start_ms = max(HISTORY_START_MS, now_ms - span_ms)
        end_ms = int(time.time() * 1000)

        run_id = await asyncio.to_thread(
            self.db.start_parse_run, symbol, timeframe, "binance-rest")

        self._emit("history_started",
                   symbol=symbol, timeframe=timeframe,
                   start_ms=start_ms, end_ms=end_ms)

        pages_box = {"pages": 0, "count": 0}

        def _progress(page, count, last_ts):
            pages_box["pages"] = page
            pages_box["count"] = count
            self._emit("history_progress",
                       symbol=symbol, timeframe=timeframe,
                       page=page, count=count, last_ts=last_ts)

        try:
            rows = await self._client.fetch_range(
                self._session, symbol, timeframe,
                start_ms, end_ms,
                progress_cb=_progress,
                sleep_sec=HISTORY_PAGE_SLEEP_SEC)
            status = "OK"
            error = None
        except Exception as e:
            rows = []
            status = "ERROR"
            error = str(e)

        await asyncio.to_thread(
            self.db.finish_parse_run, run_id,
            pages_box["pages"], len(rows), len(rows),
            status, error)

        await asyncio.to_thread(
            self.db.insert_history_request,
            symbol, timeframe, "binance-rest",
            start_ms, end_ms, pages_box["pages"], len(rows),
            status, error)

        self._emit("history_finished",
                   symbol=symbol, timeframe=timeframe,
                   total=len(rows), status=status)
        return rows

    def _update_features(self, symbol, timeframe):
        rows = self.db.get_candles_with_ids(symbol, timeframe)
        if not rows:
            return 0

        existing = self.db.get_feature_candle_ids(
            symbol, timeframe, FEATURE_SCHEMA_VERSION)

        missing_idx = [i for i, r in enumerate(rows) if r[0] not in existing]
        if not missing_idx:
            return 0

        arr = np.array([r[1:] for r in rows], dtype=float)
        f = compute_features(arr)

        out = []
        for i in missing_idx:
            cid = rows[i][0]
            out.append((
                cid,
                _v(f["return_1"][i]),
                _v(f["return_5"][i]),
                _v(f["return_20"][i]),
                _v(f["rsi"][i]),
                _v(f["ema20"][i]),
                _v(f["ema50"][i]),
                _v(f["volatility"][i]),
                _v(f["volume_ratio"][i]),
                _v(f["high_low_range"][i]),
                _v(f["body_size"][i]),
                _v(f["upper_wick"][i]),
                _v(f["lower_wick"][i]),
                FEATURE_SCHEMA_VERSION,
            ))

        return self.db.insert_features(out)

    def request_sync(self, symbol, timeframe):
        return asyncio.run_coroutine_threadsafe(
            self.sync_once(symbol, timeframe), self._loop)

    async def start_auto_sync(self, symbol, timeframe, interval_sec):
        await self.stop_auto_sync()
        self._sync_task = asyncio.create_task(
            self._auto_sync_loop(symbol, timeframe, interval_sec))

    async def stop_auto_sync(self):
        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

    async def _auto_sync_loop(self, symbol, timeframe, interval_sec):
        while True:
            try:
                await self.sync_once(symbol, timeframe)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._emit("sync_error", error=str(e))
            await asyncio.sleep(interval_sec)

    def start_auto_sync_threadsafe(self, symbol, timeframe, interval_sec):
        asyncio.run_coroutine_threadsafe(
            self.start_auto_sync(symbol, timeframe, interval_sec),
            self._loop)

    def stop_auto_sync_threadsafe(self):
        asyncio.run_coroutine_threadsafe(
            self.stop_auto_sync(), self._loop)

    def is_auto_sync_running(self):
        return self._sync_task is not None and not self._sync_task.done()

    def _load_features_matrix(self, symbol, timeframe):
        rows = self.db.get_candles_with_ids(symbol, timeframe)
        if not rows:
            return None, None

        candles = np.array([r[1:] for r in rows], dtype=float)
        n = candles.shape[0]

        feats = self.db.get_features(symbol, timeframe,
                                     FEATURE_SCHEMA_VERSION)
        fmap = {}
        for fr in feats:
            fmap[fr[0]] = np.array(
                [np.nan if v is None else float(v) for v in fr[1:]],
                dtype=float)

        X = np.full((n, len(FEATURE_COLS)), np.nan)
        for i, r in enumerate(rows):
            cid = r[0]
            v = fmap.get(cid)
            if v is not None:
                X[i] = v

        return candles, X

    async def train_async(self, symbol, timeframe, manual=True):
        from ml import train_model

        if self._train_running:
            return None
        self._train_running = True
        try:
            candles, X = await asyncio.to_thread(
                self._load_features_matrix, symbol, timeframe)
            if candles is None or X is None:
                raise ValueError("No data to train on")

            record = await asyncio.to_thread(
                train_model, symbol, timeframe, candles, X)
            await asyncio.to_thread(self.db.insert_model, record)

            max_ts = await asyncio.to_thread(
                self.db.last_timestamp, symbol, timeframe) or 0
            await asyncio.to_thread(
                self.db.set_state,
                f"last_trained_candle_ts:{symbol}:{timeframe}",
                int(max_ts))

            record["manual"] = bool(manual)
            self._emit("model_trained", **record)
            return record
        finally:
            self._train_running = False

    async def predict_async(self, symbol, timeframe, auto=False):
        from ml import load_boosters, predict_latest

        if self._predict_running:
            return None
        self._predict_running = True
        try:
            model = await asyncio.to_thread(
                self.db.get_latest_model, symbol, timeframe)
            if model is None:
                raise ValueError(
                    "No trained model for this symbol/timeframe")

            candles, X = await asyncio.to_thread(
                self._load_features_matrix, symbol, timeframe)
            if candles is None:
                return None

            cls, reg = await asyncio.to_thread(
                load_boosters, model["cls_path"], model["reg_path"])
            pred = await asyncio.to_thread(
                predict_latest, candles, X, cls, reg)

            if pred is not None:
                pred["symbol"] = symbol
                pred["timeframe"] = timeframe
                pred["model_version"] = model["version"]
                pred["horizon_seconds"] = model["horizon_seconds"]
                pred["auto"] = bool(auto)

                rows = await asyncio.to_thread(
                    self.db.get_candles_with_ids, symbol, timeframe)
                idx = pred["idx"]
                if idx < 0 or idx >= len(rows):
                    return pred
                candle_id = rows[idx][0]
                candle_ts = int(rows[idx][1])

                record = {
                    "candle_id": candle_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candle_ts": candle_ts,
                    "direction": pred["direction"],
                    "probability": float(pred["probability"]),
                    "expected_return": float(pred["expected_return"]),
                    "horizon_seconds": int(model["horizon_seconds"]),
                    "flat_threshold": float(model["flat_threshold"]),
                    "model_version": model["version"],
                    "feature_schema_ver": model.get("feature_schema_version"),
                    "created_at": int(time.time()),
                }
                pid = await asyncio.to_thread(
                    self.db.insert_prediction, record)
                pred["prediction_id"] = pid

                self._emit("prediction_made", **pred)
            return pred
        finally:
            self._predict_running = False

    async def _maybe_auto_predict(self, symbol, timeframe):
        if not AUTO_PREDICT_ON_NEW_CANDLE:
            return
        key = f"last_predict_ts:{symbol}:{timeframe}"
        last = float(await asyncio.to_thread(
            self.db.get_state, key, 0) or 0)
        if time.time() - last < AUTO_PREDICT_MIN_INTERVAL_SEC:
            return

        model = await asyncio.to_thread(
            self.db.get_latest_model, symbol, timeframe)
        if model is None:
            return

        try:
            await self.predict_async(symbol, timeframe, auto=True)
            await asyncio.to_thread(self.db.set_state, key, time.time())
        except Exception as e:
            self._emit("sync_error", error=f"auto-predict: {e}")

    async def _maybe_auto_train(self, symbol, timeframe):
        if self._train_running:
            return

        model = await asyncio.to_thread(
            self.db.get_latest_model, symbol, timeframe)
        if model is None:
            return

        if time.time() - int(model["trained_at"]) \
                < AUTO_TRAIN_MIN_INTERVAL_SEC:
            return

        key = f"last_trained_candle_ts:{symbol}:{timeframe}"
        last_ts = int(await asyncio.to_thread(
            self.db.get_state, key, 0) or 0)
        new_count = await asyncio.to_thread(
            self.db.count_candles_newer_than, symbol, timeframe, last_ts)
        if new_count < AUTO_TRAIN_MIN_NEW_CANDLES:
            return

        self._emit("auto_train_started",
                   symbol=symbol, timeframe=timeframe,
                   new_candles=new_count)
        try:
            await self.train_async(symbol, timeframe, manual=False)
        except Exception as e:
            self._emit("sync_error", error=f"auto-train: {e}")

    def set_auto_ml_enabled(self, enabled):
        self._auto_ml_enabled = bool(enabled)

    def is_auto_ml_enabled(self):
        return self._auto_ml_enabled

    def request_train(self, symbol, timeframe):
        return asyncio.run_coroutine_threadsafe(
            self.train_async(symbol, timeframe, manual=True), self._loop)

    def request_predict(self, symbol, timeframe):
        return asyncio.run_coroutine_threadsafe(
            self.predict_async(symbol, timeframe, auto=False), self._loop)

    def _load_feature_raw(self, symbol, timeframe):
        rows = self.db.get_candles_with_ids(symbol, timeframe)
        if not rows:
            return None, None, None, None

        feats = self.db.get_features(symbol, timeframe,
                                     FEATURE_SCHEMA_VERSION)
        fmap = {}
        for fr in feats:
            fmap[fr[0]] = np.array(
                [np.nan if v is None else float(v) for v in fr[1:]],
                dtype=float)

        n = len(rows)
        X_raw = np.full((n, len(FEATURE_COLS)), np.nan)
        ts_list = [0] * n
        cid_list = [0] * n
        valid = np.zeros(n, dtype=bool)

        for i, r in enumerate(rows):
            cid_list[i] = r[0]
            ts_list[i] = r[1]
            v = fmap.get(r[0])
            if v is not None:
                X_raw[i] = v
                valid[i] = True

        return X_raw, ts_list, cid_list, valid

    def _update_genomes(self, symbol, timeframe):
        if self._genome_running:
            return 0, 0
        self._genome_running = True
        try:
            X_raw, ts_list, cid_list, valid = \
                self._load_feature_raw(symbol, timeframe)
            if X_raw is None:
                return 0, 0
            if valid.sum() < GENOME_MIN_SAMPLES:
                return 0, 0

            last_processed = int(self.db.get_state(
                f"genome_last_ts:{symbol}:{timeframe}", 0) or 0)
            existing = self.db.get_genome_candle_ids(symbol, timeframe)

            if last_processed == 0:
                return self._rebuild_genomes(
                    symbol, timeframe, X_raw, ts_list, cid_list, valid)

            new_count = sum(
                1 for i in range(len(ts_list))
                if valid[i] and cid_list[i] not in existing)
            if new_count >= GENOME_REBUILD_EVERY_NEW_CANDLES:
                return self._rebuild_genomes(
                    symbol, timeframe, X_raw, ts_list, cid_list, valid)

            return self._incremental_genomes(
                symbol, timeframe, X_raw, ts_list, cid_list,
                valid, existing)
        finally:
            self._genome_running = False

    def _rebuild_genomes(self, symbol, timeframe, X_raw,
                         ts_list, cid_list, valid):
        idx_valid = np.where(valid)[0]
        Xv = X_raw[idx_valid]
        X_norm, mean, std = normalize_matrix(Xv)
        ts_v = [ts_list[i] for i in idx_valid]
        cid_v = [cid_list[i] for i in idx_valid]

        self.db.clear_genomes_for(symbol, timeframe)

        genomes, instances = build_genomes(X_norm, ts_v, cid_v)

        uid_base = self.db.get_max_genome_uid()
        now = int(time.time())

        local_to_genome_id = {}
        for g in genomes:
            uid = uid_base + g["local_idx"] + 1
            name = make_genome_name(symbol, uid)
            gid = self.db.insert_genome({
                "genome_uid": uid,
                "symbol": symbol,
                "name": name,
                "created_at": now,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            })
            local_to_genome_id[g["local_idx"]] = gid

        cid_to_local = {cid: k for k, cid in enumerate(cid_v)}

        first_instance_of_genome = {}
        for (g_idx, num, ts, cid, sim) in instances:
            k = cid_to_local.get(cid)
            if k is None:
                continue
            x_local = X_norm[k]
            gid = local_to_genome_id[g_idx]
            fp = fingerprint(x_local)
            iid = self.db.insert_genome_instance({
                "genome_id": gid,
                "instance_number": num,
                "timestamp": ts,
                "symbol": symbol,
                "timeframe": timeframe,
                "candle_id": cid,
                "fingerprint": fp,
                "similarity_to_genesis": 1.0 if num == 0 else sim,
                "feature_vector": ",".join(
                    f"{v:.6f}" for v in x_local.tolist()),
            })
            if num == 0:
                first_instance_of_genome[gid] = iid

        for gid, iid in first_instance_of_genome.items():
            self.db.update_genome_genesis(gid, iid)

        self.db.set_state(
            f"genome_mean:{symbol}:{timeframe}",
            ",".join(f"{v:.8f}" for v in mean.tolist()))
        self.db.set_state(
            f"genome_std:{symbol}:{timeframe}",
            ",".join(f"{v:.8f}" for v in std.tolist()))
        self.db.set_state(
            f"genome_last_ts:{symbol}:{timeframe}",
            int(ts_list[-1]) if ts_list else 0)

        return len(genomes), len(instances)

    def _incremental_genomes(self, symbol, timeframe, X_raw,
                             ts_list, cid_list, valid, existing):
        mean_s = self.db.get_state(
            f"genome_mean:{symbol}:{timeframe}")
        std_s = self.db.get_state(
            f"genome_std:{symbol}:{timeframe}")
        if not mean_s or not std_s:
            return self._rebuild_genomes(
                symbol, timeframe, X_raw, ts_list, cid_list, valid)

        mean = np.array([float(x) for x in mean_s.split(",")])
        std = np.array([float(x) for x in std_s.split(",")])

        genesis_rows = self.db.get_genesis_vectors(symbol, timeframe)
        if not genesis_rows:
            return self._rebuild_genomes(
                symbol, timeframe, X_raw, ts_list, cid_list, valid)

        genesis_vecs = []
        genesis_meta = []
        for (gid, uid, name, giid, fv) in genesis_rows:
            vec = np.array([float(x) for x in fv.split(",")])
            genesis_vecs.append(vec)
            genesis_meta.append({"genome_id": gid, "genesis_iid": giid})

        genesis_matrix = np.vstack(genesis_vecs)

        uid_base = self.db.get_max_genome_uid()
        now = int(time.time())
        new_genomes = 0
        new_instances = 0

        for i in range(len(ts_list)):
            if not valid[i]:
                continue
            cid = cid_list[i]
            if cid in existing:
                continue

            x_norm = normalize_vector(X_raw[i], mean, std)

            sims = genesis_matrix @ x_norm
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim >= 0.90:
                gid = genesis_meta[best_idx]["genome_id"]
                num = self.db.get_next_instance_number(gid)
                self.db.insert_genome_instance({
                    "genome_id": gid,
                    "instance_number": num,
                    "timestamp": ts_list[i],
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candle_id": cid,
                    "fingerprint": fingerprint(x_norm),
                    "similarity_to_genesis": best_sim,
                    "feature_vector": ",".join(
                        f"{v:.6f}" for v in x_norm.tolist()),
                })
                new_instances += 1
            else:
                uid = uid_base + new_genomes + 1
                name = make_genome_name(symbol, uid)
                gid = self.db.insert_genome({
                    "genome_uid": uid,
                    "symbol": symbol,
                    "name": name,
                    "created_at": now,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                })
                iid = self.db.insert_genome_instance({
                    "genome_id": gid,
                    "instance_number": 0,
                    "timestamp": ts_list[i],
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candle_id": cid,
                    "fingerprint": fingerprint(x_norm),
                    "similarity_to_genesis": 1.0,
                    "feature_vector": ",".join(
                        f"{v:.6f}" for v in x_norm.tolist()),
                })
                self.db.update_genome_genesis(gid, iid)
                genesis_matrix = np.vstack(
                    [genesis_matrix, x_norm[None, :]])
                genesis_meta.append({"genome_id": gid, "genesis_iid": iid})
                new_genomes += 1
                new_instances += 1

            existing.add(cid)

        if ts_list:
            self.db.set_state(
                f"genome_last_ts:{symbol}:{timeframe}",
                int(ts_list[-1]))

        return new_genomes, new_instances

    async def _rebuild_genomes_async(self, symbol, timeframe):
        await asyncio.to_thread(
            self.db.set_state,
            f"genome_last_ts:{symbol}:{timeframe}", 0)
        return await asyncio.to_thread(
            self._update_genomes, symbol, timeframe)

    def request_rebuild_genomes(self, symbol, timeframe):
        return asyncio.run_coroutine_threadsafe(
            self._rebuild_genomes_async(symbol, timeframe), self._loop)

    def set_proxy(self, mode, host="", port="", user="", password=""):
        self._proxy_mode = mode
        self._proxy_host = host
        self._proxy_port = port
        self._proxy_user = user
        self._proxy_pass = password

        proxy_url = None
        if mode and mode.lower() != "direct" and host and port:
            scheme = mode.lower()
            auth = ""
            if user and password:
                auth = f"{user}:{password}@"
            proxy_url = f"{scheme}://{auth}{host}:{port}"

        self._client.set_proxy(proxy_url)
        asyncio.run_coroutine_threadsafe(
            self._recreate_session(), self._loop)

    def get_proxy_state(self):
        return {
            "mode": self._proxy_mode,
            "host": self._proxy_host,
            "port": self._proxy_port,
            "user": self._proxy_user,
            "password": self._proxy_pass,
        }

    async def _recreate_session(self):
        old = self._session
        self._session = aiohttp.ClientSession()
        if old is not None:
            try:
                await old.close()
            except Exception:
                pass

    async def _ping_async(self):
        if self._session is None:
            return None, "no session"
        return await self._client.ping(self._session)

    def request_ping(self):
        return asyncio.run_coroutine_threadsafe(
            self._ping_async(), self._loop)

    async def _fill_gaps_async(self, symbol, timeframe):
        from ml import _timeframe_to_ms
        gaps = await asyncio.to_thread(
            self.db.find_gaps, symbol, timeframe, 500)
        if not gaps:
            return 0

        tf_ms = _timeframe_to_ms(timeframe)
        run_id = await asyncio.to_thread(
            self.db.start_parse_run, symbol, timeframe, "fill-gaps")

        total = 0
        try:
            for g in gaps:
                start_ms = g["from_ts"] + tf_ms
                end_ms = g["to_ts"] - tf_ms
                if start_ms > end_ms:
                    continue
                rows = await self._client.fetch_range(
                    self._session, symbol, timeframe,
                    start_ms, end_ms, sleep_sec=0.15)
                if rows:
                    n = await asyncio.to_thread(
                        self.db.insert_candles, symbol, timeframe, rows)
                    total += n
            await asyncio.to_thread(
                self.db.finish_parse_run, run_id, len(gaps),
                total, total, "OK", None)
        except Exception as e:
            await asyncio.to_thread(
                self.db.finish_parse_run, run_id, len(gaps),
                total, total, "ERROR", str(e))
            raise

        await asyncio.to_thread(
            self._update_features, symbol, timeframe)
        await asyncio.to_thread(
            self._update_genomes, symbol, timeframe)
        return total

    def request_fill_gaps(self, symbol, timeframe):
        return asyncio.run_coroutine_threadsafe(
            self._fill_gaps_async(symbol, timeframe), self._loop)

    # ---------- multi-horizon ----------

    async def train_all_horizons_async(self, symbol, timeframe):
        from ml import train_model

        if self._train_running:
            return []
        self._train_running = True
        try:
            candles, X = await asyncio.to_thread(
                self._load_features_matrix, symbol, timeframe)
            if candles is None or X is None:
                raise ValueError("No data to train on")

            records = []
            for _label, seconds in ML_HORIZONS:
                self._emit("auto_train_started",
                           symbol=symbol, timeframe=timeframe,
                           new_candles=0)
                try:
                    rec = await asyncio.to_thread(
                        train_model, symbol, timeframe,
                        candles, X,
                        ML_FLAT_THRESHOLD, seconds)
                    await asyncio.to_thread(
                        self.db.insert_model, rec)
                    rec["manual"] = True
                    self._emit("model_trained", **rec)
                    records.append(rec)
                except Exception as e:
                    self._emit("sync_error",
                               error=f"train h={seconds}: {e}")

            max_ts = await asyncio.to_thread(
                self.db.last_timestamp, symbol, timeframe) or 0
            await asyncio.to_thread(
                self.db.set_state,
                f"last_trained_candle_ts:{symbol}:{timeframe}",
                int(max_ts))
            return records
        finally:
            self._train_running = False

    def request_train_all_horizons(self, symbol, timeframe):
        return asyncio.run_coroutine_threadsafe(
            self.train_all_horizons_async(symbol, timeframe),
            self._loop)

    async def predict_all_horizons_async(self, symbol, timeframe):
        from ml import load_boosters, predict_latest

        if self._predict_running:
            return []
        self._predict_running = True
        try:
            candles, X = await asyncio.to_thread(
                self._load_features_matrix, symbol, timeframe)
            if candles is None:
                return []

            rows = await asyncio.to_thread(
                self.db.get_candles_with_ids, symbol, timeframe)

            out = []
            for label, seconds in ML_HORIZONS:
                model = await asyncio.to_thread(
                    self.db.get_latest_model, symbol, timeframe, seconds)
                if model is None:
                    continue
                try:
                    cls, reg = await asyncio.to_thread(
                        load_boosters,
                        model["cls_path"], model["reg_path"])
                    pred = await asyncio.to_thread(
                        predict_latest, candles, X, cls, reg)
                    if pred is None:
                        continue
                    pred["symbol"] = symbol
                    pred["timeframe"] = timeframe
                    pred["horizon_seconds"] = seconds
                    pred["horizon_label"] = label
                    pred["model_version"] = model["version"]
                    pred["auto"] = True

                    idx = pred["idx"]
                    if 0 <= idx < len(rows):
                        candle_id = rows[idx][0]
                        candle_ts = int(rows[idx][1])
                        record = {
                            "candle_id": candle_id,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "candle_ts": candle_ts,
                            "direction": pred["direction"],
                            "probability": float(pred["probability"]),
                            "expected_return":
                                float(pred["expected_return"]),
                            "horizon_seconds": int(seconds),
                            "flat_threshold":
                                float(model["flat_threshold"]),
                            "model_version": model["version"],
                            "feature_schema_ver":
                                model.get("feature_schema_version"),
                            "created_at": int(time.time()),
                        }
                        pid = await asyncio.to_thread(
                            self.db.insert_prediction, record)
                        pred["prediction_id"] = pid

                    out.append(pred)
                    self._emit("prediction_made", **pred)
                except Exception as e:
                    self._emit("sync_error",
                               error=f"predict h={seconds}: {e}")
            return out
        finally:
            self._predict_running = False

    def request_predict_all_horizons(self, symbol, timeframe):
        return asyncio.run_coroutine_threadsafe(
            self.predict_all_horizons_async(symbol, timeframe),
            self._loop)

    # ---------- feature importance / shap ----------

    async def feature_importance_async(self, symbol, timeframe,
                                       horizon_seconds):
        from ml import load_boosters, feature_importance

        model = await asyncio.to_thread(
            self.db.get_latest_model, symbol, timeframe, horizon_seconds)
        if model is None:
            return None
        cls, _reg = await asyncio.to_thread(
            load_boosters, model["cls_path"], model["reg_path"])
        imps = await asyncio.to_thread(feature_importance, cls)
        return {"version": model["version"], "importance": imps}

    def request_feature_importance(self, symbol, timeframe,
                                   horizon_seconds):
        return asyncio.run_coroutine_threadsafe(
            self.feature_importance_async(
                symbol, timeframe, horizon_seconds), self._loop)

    async def shap_explain_async(self, symbol, timeframe,
                                 horizon_seconds):
        from ml import load_boosters, shap_explain, FEATURE_COLS

        model = await asyncio.to_thread(
            self.db.get_latest_model, symbol, timeframe, horizon_seconds)
        if model is None:
            return None
        candles, X = await asyncio.to_thread(
            self._load_features_matrix, symbol, timeframe)
        if candles is None:
            return None
        cls, _reg = await asyncio.to_thread(
            load_boosters, model["cls_path"], model["reg_path"])
        result = await asyncio.to_thread(
            shap_explain, cls, X, FEATURE_COLS,
            SHAP_BACKGROUND_SIZE, SHAP_MAX_ROWS)
        return {"version": model["version"], "explain": result}

    def request_shap_explain(self, symbol, timeframe, horizon_seconds):
        return asyncio.run_coroutine_threadsafe(
            self.shap_explain_async(symbol, timeframe, horizon_seconds),
            self._loop)

    # ---------- consensus ----------

    async def consensus_async(self, symbol):
        from ml import load_boosters, predict_latest

        out = []
        for tf in CONSENSUS_TIMEFRAMES:
            try:
                model = await asyncio.to_thread(
                    self.db.get_latest_model, symbol, tf,
                    ML_DEFAULT_HORIZON)
                if model is None:
                    out.append({
                        "timeframe": tf,
                        "direction": "N/A",
                        "probability": 0.0,
                        "expected_return": 0.0,
                        "weight": CONSENSUS_WEIGHTS.get(tf, 1.0),
                    })
                    continue

                candles, X = await asyncio.to_thread(
                    self._load_features_matrix, symbol, tf)
                if candles is None:
                    continue
                cls, reg = await asyncio.to_thread(
                    load_boosters,
                    model["cls_path"], model["reg_path"])
                pred = await asyncio.to_thread(
                    predict_latest, candles, X, cls, reg)
                if pred is None:
                    continue
                out.append({
                    "timeframe": tf,
                    "direction": pred["direction"],
                    "probability": float(pred["probability"]),
                    "expected_return": float(pred["expected_return"]),
                    "weight": CONSENSUS_WEIGHTS.get(tf, 1.0),
                })
            except Exception:
                continue

        score = 0.0
        weight_sum = 0.0
        ret_sum = 0.0
        for r in out:
            w = r["weight"]
            ret_sum += r["expected_return"] * w
            weight_sum += w
            sign = {"UP": 1, "DOWN": -1, "FLAT": 0,
                    "N/A": 0}.get(r["direction"], 0)
            score += sign * r["probability"] * w

        if weight_sum > 0:
            norm_score = score / weight_sum
            avg_ret = ret_sum / weight_sum
        else:
            norm_score = 0.0
            avg_ret = 0.0

        if abs(norm_score) < (1.0 - CONSENSUS_MIN_CONFIDENCE):
            direction = "FLAT"
        elif norm_score > 0:
            direction = "UP"
        else:
            direction = "DOWN"

        return {
            "symbol": symbol,
            "direction": direction,
            "score": norm_score,
            "avg_expected_return": avg_ret,
            "per_timeframe": out,
        }

    def request_consensus(self, symbol):
        return asyncio.run_coroutine_threadsafe(
            self.consensus_async(symbol), self._loop)

    # ---------- calibration ----------

    async def calibration_async(self, symbol, timeframe):
        rows = await asyncio.to_thread(
            self.db.get_calibration_points, symbol, timeframe, 10)
        return rows

    def request_calibration(self, symbol, timeframe):
        return asyncio.run_coroutine_threadsafe(
            self.calibration_async(symbol, timeframe), self._loop)
        
def _v(x):
    if x is None:
        return None
    try:
        if np.isnan(x):
            return None
    except TypeError:
        pass
    return float(x)