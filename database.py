# database.py
import sqlite3
import threading
import time

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS candles (
    id        INTEGER PRIMARY KEY,
    symbol    TEXT    NOT NULL,
    timeframe TEXT    NOT NULL,
    timestamp INTEGER NOT NULL,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL    NOT NULL,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles(symbol, timeframe, timestamp);

CREATE TABLE IF NOT EXISTS features (
    candle_id       INTEGER PRIMARY KEY,
    return_1        REAL,
    return_5        REAL,
    return_20       REAL,
    rsi             REAL,
    ema20           REAL,
    ema50           REAL,
    volatility      REAL,
    volume_ratio    REAL,
    high_low_range  REAL,
    body_size       REAL,
    upper_wick      REAL,
    lower_wick      REAL,
    schema_version  TEXT,
    FOREIGN KEY(candle_id) REFERENCES candles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS models (
    id                       INTEGER PRIMARY KEY,
    version                  TEXT UNIQUE NOT NULL,
    symbol                   TEXT NOT NULL,
    timeframe                TEXT NOT NULL,
    horizon_seconds          INTEGER NOT NULL,
    flat_threshold           REAL NOT NULL,
    feature_schema_version   TEXT NOT NULL,
    n_train                  INTEGER NOT NULL,
    cls_accuracy             REAL,
    reg_mae                  REAL,
    trained_at               INTEGER NOT NULL,
    cls_path                 TEXT NOT NULL,
    reg_path                 TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_lookup
    ON models(symbol, timeframe, trained_at DESC);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY,
    candle_id           INTEGER NOT NULL,
    symbol              TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    candle_ts           INTEGER NOT NULL,
    direction           TEXT NOT NULL,
    probability         REAL NOT NULL,
    expected_return     REAL NOT NULL,
    horizon_seconds     INTEGER NOT NULL,
    flat_threshold      REAL NOT NULL,
    model_version       TEXT NOT NULL,
    feature_schema_ver  TEXT,
    genome_instance_id  INTEGER,
    created_at          INTEGER NOT NULL,
    actual_direction    TEXT,
    actual_return       REAL,
    direction_correct   INTEGER,
    return_error        REAL,
    accuracy            REAL,
    status              TEXT NOT NULL DEFAULT 'PENDING',
    resolved_at         INTEGER,
    FOREIGN KEY(candle_id) REFERENCES candles(id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_lookup
    ON predictions(symbol, timeframe, candle_ts DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_pending
    ON predictions(status, created_at);

CREATE TABLE IF NOT EXISTS genomes (
    id                     INTEGER PRIMARY KEY,
    genome_uid             INTEGER UNIQUE NOT NULL,
    symbol                 TEXT NOT NULL,
    name                   TEXT UNIQUE NOT NULL,
    genesis_instance_id    INTEGER,
    created_at             INTEGER NOT NULL,
    feature_schema_version TEXT
);

CREATE INDEX IF NOT EXISTS idx_genomes_symbol
    ON genomes(symbol);

CREATE TABLE IF NOT EXISTS genome_instances (
    id                      INTEGER PRIMARY KEY,
    genome_id               INTEGER NOT NULL,
    instance_number         INTEGER NOT NULL,
    timestamp               INTEGER NOT NULL,
    symbol                  TEXT NOT NULL,
    timeframe               TEXT NOT NULL,
    candle_id               INTEGER NOT NULL,
    fingerprint             TEXT UNIQUE NOT NULL,
    similarity_to_genesis   REAL,
    similarity_to_consensus REAL,
    feature_vector          TEXT,
    UNIQUE(genome_id, instance_number),
    FOREIGN KEY(genome_id) REFERENCES genomes(id),
    FOREIGN KEY(candle_id) REFERENCES candles(id)
);

CREATE INDEX IF NOT EXISTS idx_genome_instances_lookup
    ON genome_instances(symbol, timeframe, timestamp);

CREATE INDEX IF NOT EXISTS idx_genome_instances_genome
    ON genome_instances(genome_id, instance_number);

CREATE TABLE IF NOT EXISTS parse_runs (
    id               INTEGER PRIMARY KEY,
    started_at       INTEGER NOT NULL,
    finished_at      INTEGER,
    source           TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    timeframe        TEXT NOT NULL,
    pages            INTEGER,
    candles_received INTEGER,
    candles_inserted INTEGER,
    status           TEXT NOT NULL,
    error            TEXT
);

CREATE INDEX IF NOT EXISTS idx_parse_runs_lookup
    ON parse_runs(symbol, timeframe, started_at DESC);

CREATE TABLE IF NOT EXISTS history_requests (
    id               INTEGER PRIMARY KEY,
    requested_at     INTEGER NOT NULL,
    source           TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    timeframe        TEXT NOT NULL,
    from_ts          INTEGER NOT NULL,
    to_ts            INTEGER NOT NULL,
    pages            INTEGER,
    candles_received INTEGER,
    status           TEXT NOT NULL,
    error            TEXT
);

CREATE INDEX IF NOT EXISTS idx_history_requests_lookup
    ON history_requests(symbol, timeframe, requested_at DESC);

CREATE TABLE IF NOT EXISTS simulations (
    id              INTEGER PRIMARY KEY,
    created_at      INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    initial_capital REAL NOT NULL,
    final_capital   REAL NOT NULL,
    profit          REAL NOT NULL,
    return_pct      REAL NOT NULL,
    trades          INTEGER NOT NULL,
    wins            INTEGER NOT NULL,
    losses          INTEGER NOT NULL,
    win_rate        REAL,
    avg_trade       REAL,
    max_drawdown    REAL,
    config_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_simulations_lookup
    ON simulations(symbol, timeframe, created_at DESC);

CREATE TABLE IF NOT EXISTS simulation_trades (
    id              INTEGER PRIMARY KEY,
    simulation_id   INTEGER NOT NULL,
    prediction_id   INTEGER,
    entry_ts        INTEGER NOT NULL,
    exit_ts         INTEGER NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    direction       TEXT NOT NULL,
    size            REAL NOT NULL,
    pnl             REAL NOT NULL,
    return_pct      REAL NOT NULL,
    FOREIGN KEY(simulation_id) REFERENCES simulations(id),
    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
);

CREATE INDEX IF NOT EXISTS idx_simulation_trades_sim
    ON simulation_trades(simulation_id);
"""


class Database:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------- candles ----------

    def insert_candles(self, symbol, timeframe, rows):
        if not rows:
            return 0
        with self._lock:
            cur = self.conn.executemany(
                """INSERT OR IGNORE INTO candles
                   (symbol, timeframe, timestamp, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [(symbol, timeframe, *r) for r in rows],
            )
            self.conn.commit()
            return cur.rowcount

    def get_candles(self, symbol, timeframe, limit=None):
        with self._lock:
            rows = self.conn.execute(
                """SELECT timestamp, open, high, low, close, volume
                   FROM candles
                   WHERE symbol=? AND timeframe=?
                   ORDER BY timestamp""",
                (symbol, timeframe),
            ).fetchall()
        if limit:
            rows = rows[-limit:]
        return rows

    def get_candles_with_ids(self, symbol, timeframe):
        with self._lock:
            return self.conn.execute(
                """SELECT id, timestamp, open, high, low, close, volume
                   FROM candles
                   WHERE symbol=? AND timeframe=?
                   ORDER BY timestamp""",
                (symbol, timeframe),
            ).fetchall()

    def last_timestamp(self, symbol, timeframe):
        with self._lock:
            row = self.conn.execute(
                """SELECT MAX(timestamp) FROM candles
                   WHERE symbol=? AND timeframe=?""",
                (symbol, timeframe),
            ).fetchone()
        return row[0] if row and row[0] else None

    def first_timestamp(self, symbol, timeframe):
        with self._lock:
            row = self.conn.execute(
                """SELECT MIN(timestamp) FROM candles
                   WHERE symbol=? AND timeframe=?""",
                (symbol, timeframe),
            ).fetchone()
        return row[0] if row and row[0] else None

    def get_candle_close_at(self, symbol, timeframe, ts):
        with self._lock:
            row = self.conn.execute(
                """SELECT close FROM candles
                   WHERE symbol=? AND timeframe=? AND timestamp=?""",
                (symbol, timeframe, ts),
            ).fetchone()
        return row[0] if row else None

    def count_candles_newer_than(self, symbol, timeframe, ts):
        with self._lock:
            row = self.conn.execute(
                """SELECT COUNT(*) FROM candles
                   WHERE symbol=? AND timeframe=? AND timestamp>?""",
                (symbol, timeframe, int(ts)),
            ).fetchone()
        return row[0] if row else 0

    def count_candles(self, symbol, timeframe):
        with self._lock:
            row = self.conn.execute(
                """SELECT COUNT(*) FROM candles
                   WHERE symbol=? AND timeframe=?""",
                (symbol, timeframe),
            ).fetchone()
        return row[0] if row else 0

    # ---------- features ----------

    def get_feature_candle_ids(self, symbol, timeframe, schema_version):
        with self._lock:
            rows = self.conn.execute(
                """SELECT f.candle_id
                   FROM features f
                   JOIN candles c ON c.id = f.candle_id
                   WHERE c.symbol=? AND c.timeframe=?
                     AND f.schema_version=?""",
                (symbol, timeframe, schema_version),
            ).fetchall()
        return {r[0] for r in rows}

    def insert_features(self, rows):
        if not rows:
            return 0
        with self._lock:
            cur = self.conn.executemany(
                """INSERT OR REPLACE INTO features
                   (candle_id,
                    return_1, return_5, return_20,
                    rsi, ema20, ema50,
                    volatility, volume_ratio,
                    high_low_range, body_size, upper_wick, lower_wick,
                    schema_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self.conn.commit()
            return cur.rowcount

    def count_features(self, symbol, timeframe):
        with self._lock:
            row = self.conn.execute(
                """SELECT COUNT(*)
                   FROM features f
                   JOIN candles c ON c.id = f.candle_id
                   WHERE c.symbol=? AND c.timeframe=?""",
                (symbol, timeframe),
            ).fetchone()
        return row[0] if row else 0

    def get_features(self, symbol, timeframe, schema_version):
        with self._lock:
            return self.conn.execute(
                """SELECT f.candle_id,
                          f.return_1, f.return_5, f.return_20,
                          f.rsi, f.ema20, f.ema50,
                          f.volatility, f.volume_ratio,
                          f.high_low_range, f.body_size,
                          f.upper_wick, f.lower_wick
                   FROM features f
                   JOIN candles c ON c.id = f.candle_id
                   WHERE c.symbol=? AND c.timeframe=?
                     AND f.schema_version=?
                   ORDER BY c.timestamp""",
                (symbol, timeframe, schema_version),
            ).fetchall()

    # ---------- models ----------

    def insert_model(self, r):
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO models
                   (version, symbol, timeframe, horizon_seconds,
                    flat_threshold, feature_schema_version,
                    n_train, cls_accuracy, reg_mae,
                    trained_at, cls_path, reg_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (r["version"], r["symbol"], r["timeframe"],
                 r["horizon_seconds"], r["flat_threshold"],
                 r["feature_schema_version"],
                 r["n_train"], r["cls_accuracy"], r["reg_mae"],
                 r["trained_at"], r["cls_path"], r["reg_path"]),
            )
            self.conn.commit()

    def get_latest_model(self, symbol, timeframe, horizon_seconds=None):
        q = """SELECT version, symbol, timeframe, horizon_seconds,
                      flat_threshold, feature_schema_version,
                      n_train, cls_accuracy, reg_mae,
                      trained_at, cls_path, reg_path
               FROM models
               WHERE symbol=? AND timeframe=?"""
        args = [symbol, timeframe]
        if horizon_seconds is not None:
            q += " AND horizon_seconds=?"
            args.append(int(horizon_seconds))
        q += " ORDER BY trained_at DESC LIMIT 1"
        with self._lock:
            row = self.conn.execute(q, args).fetchone()
        if not row:
            return None
        keys = ["version", "symbol", "timeframe", "horizon_seconds",
                "flat_threshold", "feature_schema_version",
                "n_train", "cls_accuracy", "reg_mae",
                "trained_at", "cls_path", "reg_path"]
        return dict(zip(keys, row))

    def get_all_models(self, symbol, timeframe):
        with self._lock:
            rows = self.conn.execute(
                """SELECT version, symbol, timeframe, horizon_seconds,
                          flat_threshold, feature_schema_version,
                          n_train, cls_accuracy, reg_mae,
                          trained_at, cls_path, reg_path
                   FROM models
                   WHERE symbol=? AND timeframe=?
                   ORDER BY trained_at DESC""",
                (symbol, timeframe),
            ).fetchall()
        keys = ["version", "symbol", "timeframe", "horizon_seconds",
                "flat_threshold", "feature_schema_version",
                "n_train", "cls_accuracy", "reg_mae",
                "trained_at", "cls_path", "reg_path"]
        return [dict(zip(keys, r)) for r in rows]

    # ---------- predictions ----------

    def insert_prediction(self, r):
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO predictions
                   (candle_id, symbol, timeframe, candle_ts,
                    direction, probability, expected_return,
                    horizon_seconds, flat_threshold,
                    model_version, feature_schema_ver,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (r["candle_id"], r["symbol"], r["timeframe"], r["candle_ts"],
                 r["direction"], r["probability"], r["expected_return"],
                 r["horizon_seconds"], r["flat_threshold"],
                 r["model_version"], r.get("feature_schema_ver"),
                 r["created_at"]),
            )
            self.conn.commit()
            return cur.lastrowid

    def get_pending_predictions(self):
        with self._lock:
            rows = self.conn.execute(
                """SELECT id, candle_id, symbol, timeframe, candle_ts,
                          direction, expected_return, horizon_seconds,
                          flat_threshold, created_at
                   FROM predictions
                   WHERE status='PENDING'
                   ORDER BY created_at""",
            ).fetchall()
        keys = ["id", "candle_id", "symbol", "timeframe", "candle_ts",
                "direction", "expected_return", "horizon_seconds",
                "flat_threshold", "created_at"]
        return [dict(zip(keys, r)) for r in rows]

    def get_prediction_probability(self, pid):
        with self._lock:
            row = self.conn.execute(
                "SELECT probability FROM predictions WHERE id=?",
                (pid,),
            ).fetchone()
        return row[0] if row else None

    def resolve_prediction(self, pid, actual_dir, actual_ret,
                           dir_correct, ret_err, accuracy, resolved_at):
        with self._lock:
            self.conn.execute(
                """UPDATE predictions
                   SET actual_direction=?, actual_return=?,
                       direction_correct=?, return_error=?, accuracy=?,
                       status='RESOLVED', resolved_at=?
                   WHERE id=?""",
                (actual_dir, actual_ret, dir_correct,
                 ret_err, accuracy, resolved_at, pid),
            )
            self.conn.commit()

    def get_predictions(self, symbol=None, timeframe=None,
                        status=None, limit=500):
        q = """SELECT id, candle_id, symbol, timeframe, candle_ts,
                      direction, probability, expected_return,
                      horizon_seconds, flat_threshold,
                      model_version, created_at,
                      actual_direction, actual_return,
                      direction_correct, return_error, accuracy,
                      status, resolved_at
               FROM predictions WHERE 1=1"""
        args = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        if timeframe:
            q += " AND timeframe=?"
            args.append(timeframe)
        if status:
            q += " AND status=?"
            args.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        keys = ["id", "candle_id", "symbol", "timeframe", "candle_ts",
                "direction", "probability", "expected_return",
                "horizon_seconds", "flat_threshold",
                "model_version", "created_at",
                "actual_direction", "actual_return",
                "direction_correct", "return_error", "accuracy",
                "status", "resolved_at"]
        return [dict(zip(keys, r)) for r in rows]

    def prediction_stats(self, symbol=None, timeframe=None):
        q = """SELECT
                 COUNT(*) AS total,
                 SUM(CASE WHEN status='RESOLVED' THEN 1 ELSE 0 END) AS resolved,
                 SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) AS pending,
                 SUM(CASE WHEN direction_correct=1 THEN 1 ELSE 0 END) AS hits,
                 AVG(CASE WHEN status='RESOLVED' THEN return_error END) AS mae,
                 AVG(CASE WHEN status='RESOLVED' THEN accuracy END) AS mean_acc
               FROM predictions WHERE 1=1"""
        args = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        if timeframe:
            q += " AND timeframe=?"
            args.append(timeframe)
        with self._lock:
            row = self.conn.execute(q, args).fetchone()
        if not row:
            return None
        keys = ["total", "resolved", "pending", "hits", "mae", "mean_acc"]
        return dict(zip(keys, row))

    # ---------- state ----------

    def get_state(self, key, default=None):
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM state WHERE key=?", (key,)
            ).fetchone()
        return row[0] if row else default

    def set_state(self, key, value):
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO state(key, value) VALUES (?, ?)",
                (key, str(value)),
            )
            self.conn.commit()

    # ---------- genomes ----------

    def get_max_genome_uid(self):
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(genome_uid), 0) FROM genomes"
            ).fetchone()
        return int(row[0]) if row else 0

    def insert_genome(self, r):
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO genomes
                   (genome_uid, symbol, name, created_at,
                    feature_schema_version)
                   VALUES (?, ?, ?, ?, ?)""",
                (r["genome_uid"], r["symbol"], r["name"],
                 r["created_at"], r.get("feature_schema_version")),
            )
            self.conn.commit()
            return cur.lastrowid

    def update_genome_genesis(self, genome_id, genesis_instance_id):
        with self._lock:
            self.conn.execute(
                "UPDATE genomes SET genesis_instance_id=? WHERE id=?",
                (genesis_instance_id, genome_id),
            )
            self.conn.commit()

    def insert_genome_instance(self, r):
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO genome_instances
                   (genome_id, instance_number, timestamp,
                    symbol, timeframe, candle_id, fingerprint,
                    similarity_to_genesis, similarity_to_consensus,
                    feature_vector)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (r["genome_id"], r["instance_number"], r["timestamp"],
                 r["symbol"], r["timeframe"], r["candle_id"],
                 r["fingerprint"],
                 r.get("similarity_to_genesis"),
                 r.get("similarity_to_consensus"),
                 r.get("feature_vector")),
            )
            self.conn.commit()
            return cur.lastrowid

    def get_next_instance_number(self, genome_id):
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(instance_number), -1) "
                "FROM genome_instances WHERE genome_id=?",
                (genome_id,),
            ).fetchone()
        return int(row[0]) + 1 if row else 0

    def get_genome_candle_ids(self, symbol, timeframe):
        with self._lock:
            rows = self.conn.execute(
                """SELECT candle_id FROM genome_instances
                   WHERE symbol=? AND timeframe=?""",
                (symbol, timeframe),
            ).fetchall()
        return {r[0] for r in rows}

    def get_genesis_vectors(self, symbol, timeframe):
        with self._lock:
            return self.conn.execute(
                """SELECT g.id, g.genome_uid, g.name,
                          gi.id, gi.feature_vector
                   FROM genomes g
                   JOIN genome_instances gi
                     ON gi.id = g.genesis_instance_id
                   WHERE g.symbol=? AND gi.timeframe=?""",
                (symbol, timeframe),
            ).fetchall()

    def get_genomes(self, symbol=None):
        q = """SELECT id, genome_uid, symbol, name,
                      genesis_instance_id, created_at,
                      feature_schema_version
               FROM genomes WHERE 1=1"""
        args = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        q += " ORDER BY genome_uid"
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        keys = ["id", "genome_uid", "symbol", "name",
                "genesis_instance_id", "created_at",
                "feature_schema_version"]
        return [dict(zip(keys, r)) for r in rows]

    def get_genome_instances(self, genome_id):
        with self._lock:
            rows = self.conn.execute(
                """SELECT id, genome_id, instance_number, timestamp,
                          symbol, timeframe, candle_id, fingerprint,
                          similarity_to_genesis, similarity_to_consensus,
                          feature_vector
                   FROM genome_instances
                   WHERE genome_id=?
                   ORDER BY instance_number""",
                (genome_id,),
            ).fetchall()
        keys = ["id", "genome_id", "instance_number", "timestamp",
                "symbol", "timeframe", "candle_id", "fingerprint",
                "similarity_to_genesis", "similarity_to_consensus",
                "feature_vector"]
        return [dict(zip(keys, r)) for r in rows]

    def count_genomes(self, symbol=None):
        q = "SELECT COUNT(*) FROM genomes"
        args = []
        if symbol:
            q += " WHERE symbol=?"
            args.append(symbol)
        with self._lock:
            row = self.conn.execute(q, args).fetchone()
        return row[0] if row else 0

    def count_genome_instances(self, symbol=None, timeframe=None):
        q = "SELECT COUNT(*) FROM genome_instances WHERE 1=1"
        args = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        if timeframe:
            q += " AND timeframe=?"
            args.append(timeframe)
        with self._lock:
            row = self.conn.execute(q, args).fetchone()
        return row[0] if row else 0

    def clear_genomes_for(self, symbol, timeframe):
        with self._lock:
            self.conn.execute(
                """DELETE FROM genome_instances
                   WHERE symbol=? AND timeframe=?""",
                (symbol, timeframe),
            )
            self.conn.execute(
                """DELETE FROM genomes
                   WHERE symbol=? AND id NOT IN (
                       SELECT DISTINCT genome_id FROM genome_instances
                   )""",
                (symbol,),
            )
            self.conn.commit()

    def get_genome_by_id(self, genome_id):
        with self._lock:
            row = self.conn.execute(
                """SELECT id, genome_uid, symbol, name,
                          genesis_instance_id, created_at,
                          feature_schema_version
                   FROM genomes WHERE id=?""",
                (genome_id,),
            ).fetchone()
        if not row:
            return None
        keys = ["id", "genome_uid", "symbol", "name",
                "genesis_instance_id", "created_at",
                "feature_schema_version"]
        return dict(zip(keys, row))

    def get_genome_instance_by_id(self, instance_id):
        with self._lock:
            row = self.conn.execute(
                """SELECT id, genome_id, instance_number, timestamp,
                          symbol, timeframe, candle_id, fingerprint,
                          similarity_to_genesis, similarity_to_consensus,
                          feature_vector
                   FROM genome_instances WHERE id=?""",
                (instance_id,),
            ).fetchone()
        if not row:
            return None
        keys = ["id", "genome_id", "instance_number", "timestamp",
                "symbol", "timeframe", "candle_id", "fingerprint",
                "similarity_to_genesis", "similarity_to_consensus",
                "feature_vector"]
        return dict(zip(keys, row))

    def get_genome_stats(self, genome_id):
        with self._lock:
            row = self.conn.execute(
                """SELECT
                     COUNT(*) AS total,
                     AVG(similarity_to_genesis) AS avg_sim
                   FROM genome_instances WHERE genome_id=?""",
                (genome_id,),
            ).fetchone()
        return {
            "total": row[0] if row else 0,
            "avg_sim": row[1] if row and row[1] is not None else None,
        }

    # ---------- parse runs ----------

    def start_parse_run(self, symbol, timeframe, source):
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO parse_runs
                   (started_at, source, symbol, timeframe, status)
                   VALUES (?, ?, ?, ?, 'RUNNING')""",
                (int(time.time()), source, symbol, timeframe),
            )
            self.conn.commit()
            return cur.lastrowid

    def finish_parse_run(self, run_id, pages, received, inserted,
                         status="OK", error=None):
        with self._lock:
            self.conn.execute(
                """UPDATE parse_runs
                   SET finished_at=?, pages=?, candles_received=?,
                       candles_inserted=?, status=?, error=?
                   WHERE id=?""",
                (int(time.time()), pages, received, inserted,
                 status, error, run_id),
            )
            self.conn.commit()

    def get_parse_runs(self, symbol=None, limit=200):
        q = """SELECT id, started_at, finished_at, source,
                      symbol, timeframe, pages,
                      candles_received, candles_inserted,
                      status, error
               FROM parse_runs WHERE 1=1"""
        args = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        q += " ORDER BY started_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        keys = ["id", "started_at", "finished_at", "source",
                "symbol", "timeframe", "pages",
                "candles_received", "candles_inserted",
                "status", "error"]
        return [dict(zip(keys, r)) for r in rows]

    # ---------- history requests ----------

    def insert_history_request(self, symbol, timeframe, source,
                               from_ts, to_ts, pages, received,
                               status="OK", error=None):
        with self._lock:
            self.conn.execute(
                """INSERT INTO history_requests
                   (requested_at, source, symbol, timeframe,
                    from_ts, to_ts, pages, candles_received,
                    status, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (int(time.time()), source, symbol, timeframe,
                 from_ts, to_ts, pages, received, status, error),
            )
            self.conn.commit()

    def get_history_requests(self, symbol=None, limit=200):
        q = """SELECT id, requested_at, source, symbol, timeframe,
                      from_ts, to_ts, pages, candles_received,
                      status, error
               FROM history_requests WHERE 1=1"""
        args = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        q += " ORDER BY requested_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        keys = ["id", "requested_at", "source", "symbol", "timeframe",
                "from_ts", "to_ts", "pages", "candles_received",
                "status", "error"]
        return [dict(zip(keys, r)) for r in rows]

    # ---------- simulations ----------

    def insert_simulation(self, r):
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO simulations
                   (created_at, symbol, timeframe,
                    initial_capital, final_capital, profit, return_pct,
                    trades, wins, losses, win_rate, avg_trade,
                    max_drawdown, config_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (r["created_at"], r["symbol"], r["timeframe"],
                 r["initial_capital"], r["final_capital"],
                 r["profit"], r["return_pct"],
                 r["trades"], r["wins"], r["losses"],
                 r["win_rate"], r["avg_trade"],
                 r["max_drawdown"], r["config_json"]),
            )
            self.conn.commit()
            return cur.lastrowid

    def insert_simulation_trades(self, rows):
        if not rows:
            return 0
        with self._lock:
            cur = self.conn.executemany(
                """INSERT INTO simulation_trades
                   (simulation_id, prediction_id, entry_ts, exit_ts,
                    entry_price, exit_price, direction, size,
                    pnl, return_pct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            self.conn.commit()
            return cur.rowcount

    def get_simulations(self, symbol=None, timeframe=None, limit=100):
        q = """SELECT id, created_at, symbol, timeframe,
                      initial_capital, final_capital, profit, return_pct,
                      trades, wins, losses, win_rate, avg_trade,
                      max_drawdown, config_json
               FROM simulations WHERE 1=1"""
        args = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        if timeframe:
            q += " AND timeframe=?"
            args.append(timeframe)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        keys = ["id", "created_at", "symbol", "timeframe",
                "initial_capital", "final_capital", "profit", "return_pct",
                "trades", "wins", "losses", "win_rate", "avg_trade",
                "max_drawdown", "config_json"]
        return [dict(zip(keys, r)) for r in rows]

    def get_simulation_trades(self, sim_id):
        with self._lock:
            rows = self.conn.execute(
                """SELECT id, simulation_id, prediction_id,
                          entry_ts, exit_ts, entry_price, exit_price,
                          direction, size, pnl, return_pct
                   FROM simulation_trades
                   WHERE simulation_id=?
                   ORDER BY entry_ts""",
                (sim_id,),
            ).fetchall()
        keys = ["id", "simulation_id", "prediction_id",
                "entry_ts", "exit_ts", "entry_price", "exit_price",
                "direction", "size", "pnl", "return_pct"]
        return [dict(zip(keys, r)) for r in rows]

    # ---------- database health / coverage ----------

    def table_stats(self):
        with self._lock:
            names = [r[0] for r in self.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name").fetchall()]
            out = []
            for name in names:
                try:
                    rows = self.conn.execute(
                        f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                except Exception:
                    rows = 0
                out.append({"name": name, "rows": rows})
            return out

    def table_sizes(self):
        with self._lock:
            has = False
            try:
                self.conn.execute(
                    "SELECT COUNT(*) FROM dbstat").fetchone()
                has = True
            except Exception:
                has = False
            if not has:
                return {}
            try:
                rows = self.conn.execute(
                    """SELECT name, SUM(pgsize)
                       FROM dbstat
                       GROUP BY name""").fetchall()
            except Exception:
                return {}
            return {r[0]: int(r[1]) for r in rows}

    def has_dbstat(self):
        with self._lock:
            try:
                self.conn.execute(
                    "SELECT COUNT(*) FROM dbstat").fetchone()
                return True
            except Exception:
                return False

    def integrity_check(self):
        with self._lock:
            try:
                row = self.conn.execute(
                    "PRAGMA integrity_check").fetchone()
                return row[0] if row else "unknown"
            except Exception as e:
                return f"error: {e}"

    def quick_check(self):
        with self._lock:
            try:
                row = self.conn.execute(
                    "PRAGMA quick_check").fetchone()
                return row[0] if row else "unknown"
            except Exception as e:
                return f"error: {e}"

    def vacuum(self):
        with self._lock:
            try:
                self.conn.execute("VACUUM")
                self.conn.commit()
                return True, None
            except Exception as e:
                return False, str(e)

    def find_duplicate_candles(self, limit=50):
        with self._lock:
            rows = self.conn.execute(
                """SELECT symbol, timeframe, timestamp, COUNT(*) as c
                   FROM candles
                   GROUP BY symbol, timeframe, timestamp
                   HAVING c > 1
                   ORDER BY c DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [{"symbol": r[0], "timeframe": r[1],
                 "timestamp": r[2], "count": r[3]} for r in rows]

    def find_orphans(self):
        with self._lock:
            feat = self.conn.execute(
                """SELECT COUNT(*) FROM features f
                   LEFT JOIN candles c ON c.id = f.candle_id
                   WHERE c.id IS NULL""").fetchone()[0]
            pred = self.conn.execute(
                """SELECT COUNT(*) FROM predictions p
                   LEFT JOIN candles c ON c.id = p.candle_id
                   WHERE c.id IS NULL""").fetchone()[0]
            gi = self.conn.execute(
                """SELECT COUNT(*) FROM genome_instances gi
                   LEFT JOIN candles c ON c.id = gi.candle_id
                   WHERE c.id IS NULL""").fetchone()[0]
        return {"features": feat, "predictions": pred,
                "genome_instances": gi}

    def coverage_summary(self):
        with self._lock:
            rows = self.conn.execute(
                """SELECT symbol, timeframe, COUNT(*),
                          MIN(timestamp), MAX(timestamp)
                   FROM candles
                   GROUP BY symbol, timeframe
                   ORDER BY symbol, timeframe""").fetchall()
        out = []
        for r in rows:
            out.append({
                "symbol": r[0],
                "timeframe": r[1],
                "count": r[2],
                "first": r[3],
                "last": r[4],
            })
        return out

    @staticmethod
    def _tf_ms(timeframe):
        unit = timeframe[-1]
        n = int(timeframe[:-1])
        return {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit] * n

    def find_gaps(self, symbol, timeframe, limit=500):
        tf = self._tf_ms(timeframe)
        with self._lock:
            rows = self.conn.execute(
                """SELECT timestamp FROM candles
                   WHERE symbol=? AND timeframe=?
                   ORDER BY timestamp""",
                (symbol, timeframe),
            ).fetchall()
        gaps = []
        for i in range(1, len(rows)):
            diff = rows[i][0] - rows[i - 1][0]
            if diff > tf:
                missing = diff // tf - 1
                gaps.append({
                    "from_ts": rows[i - 1][0],
                    "to_ts": rows[i][0],
                    "missing": int(missing),
                })
                if len(gaps) >= limit:
                    break
        return gaps

    def overview(self):
        with self._lock:
            total_rows = 0
            tables = self.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for (name,) in tables:
                try:
                    total_rows += self.conn.execute(
                        f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                except Exception:
                    pass

            oldest = self.conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM candles"
            ).fetchone()

            counts = {}
            for name in ("candles", "features", "predictions",
                         "genomes", "genome_instances", "models",
                         "parse_runs", "history_requests",
                         "simulations", "simulation_trades"):
                try:
                    counts[name] = self.conn.execute(
                        f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                except Exception:
                    counts[name] = 0

            pending = self.conn.execute(
                "SELECT COUNT(*) FROM predictions "
                "WHERE status='PENDING'").fetchone()[0]
            resolved = self.conn.execute(
                "SELECT COUNT(*) FROM predictions "
                "WHERE status='RESOLVED'").fetchone()[0]

        return {
            "tables": len(tables),
            "total_rows": total_rows,
            "oldest": oldest[0] if oldest else None,
            "newest": oldest[1] if oldest else None,
            "counts": counts,
            "pending": pending,
            "resolved": resolved,
        }

    # ---------- insights ----------

    def get_calibration_points(self, symbol=None, timeframe=None,
                               buckets=10):
        q = """SELECT probability, direction_correct
               FROM predictions
               WHERE status='RESOLVED'"""
        args = []
        if symbol:
            q += " AND symbol=?"
            args.append(symbol)
        if timeframe:
            q += " AND timeframe=?"
            args.append(timeframe)
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        if not rows:
            return []

        bucket_data = [[] for _ in range(buckets)]
        for prob, correct in rows:
            if prob is None or correct is None:
                continue
            b = min(int(prob * buckets), buckets - 1)
            bucket_data[b].append(int(correct))

        out = []
        for i, hits in enumerate(bucket_data):
            if not hits:
                continue
            avg_p = (i + 0.5) / buckets
            out.append({
                "bucket": i,
                "predicted": avg_p,
                "actual": sum(hits) / len(hits),
                "count": len(hits),
            })
        return out
           
    # ---------- misc ----------

    def close(self):
        with self._lock:
            self.conn.close()