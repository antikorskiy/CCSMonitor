# README.md

```markdown
# CCSMonitor

**Local open-source cryptocurrency market analyzer with ML forecasts, Genome clustering, and full prediction history.**

---

## English

### Overview

CCSMonitor is a local open-source cryptocurrency market analyzer written in Python. It collects OHLCV history from Binance, engineers features, trains LightGBM models on multiple horizons with walk-forward validation, clusters similar market states into Genomes, and stores every prediction to later compare it with reality.

**Not a trading bot.** No order execution, no automated trading. This is a research tool for measuring its own accuracy on historical data.

### Philosophy

The system should not just say what the market might do. It stores the state of the market at the moment of prediction, what the model predicted, which genome that state belonged to — and then what actually happened.

Months or years later, you can open any candle and reconstruct:

```
Market state
+ Features
+ Genome
+ Prediction
+ Model version
+ Actual result
```

### Features

**Data & Storage**
- Async OHLCV loader from Binance REST (`aiohttp`)
- Local SQLite storage (WAL mode), multi-symbol, multi-timeframe
- Auto-sync with configurable interval
- Automatic gap detection and fill (`Coverage` window)
- Proxy support: DIRECT / HTTP / HTTPS / SOCKS4 / SOCKS5
- Network Monitor: latency, throughput, requests/errors/retries, 429 handling
- Auto-reconnect with exponential backoff

**Features (12 per candle)**
- `return_1`, `return_5`, `return_20`
- `rsi`, `ema20`, `ema50`
- `volatility`, `volume_ratio`
- `high_low_range`, `body_size`, `upper_wick`, `lower_wick`

**Machine Learning**
- LightGBM classifier (`UP` / `FLAT` / `DOWN`) and regressor (`expected_return`)
- Walk-forward validation via `TimeSeriesSplit` — no data leakage
- Multi-horizon: 5m / 15m / 1h / 4h / 24h
- Isotonic probability calibration
- Feature importance and SHAP explanations
- Multi-timeframe consensus (5m + 15m + 1h)
- Automatic retraining when new data accumulates

**Prediction History**
- Every prediction is stored in SQLite
- `PENDING → RESOLVED` lifecycle
- Automatic resolver after the horizon is reached
- Real statistics: hit-rate, MAE, calibration buckets
- Historical prediction arrows on the chart (green = correct, red = wrong, grey = pending)

**Genome Engine**
- Cosine similarity clustering of normalized feature vectors
- `Genesis #0` — first reference exemplar, never replaced
- Instances — all subsequent matches of the same family
- Automatic creation of new genomes when similarity is too low
- Genome tree UI: `CCSMonitor → Coin → Genome → Genesis → Instances`
- Fingerprint (SHA1) for each instance

**Simulation**
- Capital simulation over resolved predictions
- Stop-loss / take-profit
- Position sizing: fixed / percent / Kelly
- Equity curve visualization
- Per-trade table with PnL and reason (SL / TP / horizon)

**UI (Tkinter + Matplotlib)**
- Dark theme, flow toolbar
- Candlestick chart with EMA20/EMA50, RSI panel
- Zoom under cursor, pan, crosshair, hover tooltip
- Prediction arrows on both panels
- Genome tree, Node Inspector
- Predictions window, Logs window, Database window
- Pipeline, Coverage, Storage Monitor, ML Insights, Consensus
- Tray icon (pystray), hotkeys, system alerts

**Export**
- CSV, XLSX, JSON
- Tables: candles, features, predictions, models, genomes, genome_instances, parse_runs, history_requests, simulations, simulation_trades

### Stack

```
Python 3.10+
├── SQLite3 (WAL)
├── aiohttp (Binance REST)
├── LightGBM (classifier + regressor)
├── scikit-learn (walk-forward, calibration)
├── SHAP (explanations)
├── NumPy, pandas
├── Tkinter + Matplotlib (UI)
├── pystray + Pillow (tray)
└── openpyxl (XLSX)
```

PyTorch is not used — MVP works with tabular features and LightGBM.

### Installation

```bash
git clone https://github.com/CCSMonitor
cd CCSMonitor
python -m pip install -r requirements.txt
python app.py
```

### Requirements

- Python 3.10+
- Access to Binance API (region-dependent)
- ~50 MB for DB, models and logs initially (grows with history)

### Usage

**Basic flow**

1. Run `app.py`.
2. Select asset and timeframe, choose history window (`1y / 2y / 3y / 5y / Full`).
3. Click `LOAD / SYNC`.
4. After data is loaded, click `Train ML` (once).
5. From now on, `Auto ML` handles retraining and prediction automatically.
6. Click `Predict` / `Predict All` for manual forecasts.
7. Open `Predictions` to see real hit-rate and MAE.
8. Open `Consensus` for multi-timeframe signal.
9. Open `Simulate` to test a strategy with SL/TP and position sizing.

**Hotkeys**

| Key | Action |
|---|---|
| `Ctrl+L` | Load / sync |
| `Ctrl+R` | Sync now |
| `Ctrl+T` | Train ML |
| `Ctrl+P` | Predict |
| `Ctrl+Shift+P` | Predict all horizons |
| `Ctrl+Shift+C` | Consensus |
| `Ctrl+G` | Genomes |
| `Ctrl+I` | ML Insights |
| `Ctrl+S` | Simulate |
| `Ctrl+E` | Export |
| `Ctrl+0` | Reset zoom |
| `+` / `−` | Zoom in / out |
| `F1`–`F6` | Network / Proxy / Pipeline / Coverage / Database / Logs |
| `Esc` | Close top window |

### Configuration

Key parameters in `config.py`:

```python
ML_HORIZON_SECONDS = 3600          # default prediction horizon
ML_FLAT_THRESHOLD = 0.003          # ±0.3% boundary for UP/DOWN vs FLAT
ML_HORIZONS = [("5m", 300), ("15m", 900), ("1h", 3600), ("4h", 14400), ("24h", 86400)]

AUTO_TRAIN_MIN_INTERVAL_SEC = 6 * 3600     # min 6h between retrainings
AUTO_TRAIN_MIN_NEW_CANDLES = 500           # min new candles to retrain

GENOME_SIMILARITY_THRESHOLD = 0.90         # cosine similarity threshold
CONSENSUS_TIMEFRAMES = ["5m", "15m", "1h"]

SIMULATION_STOP_LOSS_PCT = 0.02
SIMULATION_TAKE_PROFIT_PCT = 0.04
SIMULATION_SIZING_MODE = "fixed"           # fixed | percent | kelly
```

### Anti-Leakage Principles

- X[i] uses only past and current data.
- y[i] = `close[i+h]/close[i] − 1` — target, never in X.
- Train on first `n − h` candles (no target looks past the series).
- Walk-forward via `TimeSeriesSplit`, no shuffling.
- Prediction uses the last candle with a full feature vector, not a random one.
- Normalization fitted only on training data during validation.

### What it does NOT do

- No order execution (not a trading bot).
- No WebSocket (REST polling with configurable interval).
- No Parquet storage (everything in SQLite).
- No "buy/sell" signals — only statistical probabilities and accuracy measurements.
- No guarantee of accuracy or profitability.

### Project Structure

```
CCSMonitor/
├── app.py                # entry point
├── config.py             # all constants
├── database.py           # SQLite + schema + methods
├── market.py             # async Binance client
├── async_engine.py       # event loop, sync, ML, genome, resolver
├── features.py           # 12 features
├── ml.py                 # LightGBM, walk-forward, calibration, SHAP
├── genome.py             # clustering, fingerprint, genome naming
├── network.py            # NetworkStats
├── storage.py            # disk usage reporting
├── simulation.py         # capital / SL / TP / sizing
├── export.py             # CSV / XLSX / JSON
├── tray.py               # tray icon (pystray)
├── alerts.py             # notifications
├── ui.py                 # Tkinter + Matplotlib UI
├── schema.sql            # (embedded in database.py)
├── requirements.txt
├── data/                 # optional Parquet (not used yet)
├── models/               # trained boosters
├── logs/                 # optional file logs
└── exports/              # exported tables
```

### Roadmap

**Done**
- [x] Data Core (REST + SQLite + WAL)
- [x] Graphs (candles + EMA + RSI)
- [x] Feature Engine (12 features)
- [x] ML (LightGBM + walk-forward)
- [x] Prediction History + Resolver
- [x] Genome Engine + Genome UI
- [x] Full history load (1y–5y–Full)
- [x] Network Monitor, Proxy UI, Auto-reconnect, 429
- [x] Pipeline UI, Coverage, Database Health, Storage Monitor
- [x] Feature importance, SHAP, calibration
- [x] Multi-horizon, multi-timeframe consensus
- [x] Equity curve, SL/TP, position sizing
- [x] Logs, CSV/XLSX/JSON export, simulation
- [x] Hotkeys, tray icon, alerts

### License

GNU GPL v3.

### Disclaimer

This software is provided for **research and educational purposes only**. It is not financial advice. Cryptocurrency markets are highly volatile. Any use of this software for real trading is at your own risk. The authors take no responsibility for any financial losses.

---

## Русский

### Обзор

CCSMonitor — локальный open-source анализатор рынка криптовалют на Python. Собирает историю OHLCV с Binance, считает признаки, обучает LightGBM-модели на нескольких горизонтах с walk-forward валидацией, группирует похожие состояния рынка в Genome Engine и сохраняет каждый прогноз для последующего сравнения с реальностью.

**Не торговый бот.** Нет исполнения сделок, нет автоматической торговли. Это исследовательский инструмент для измерения собственной точности на исторических данных.

### Философия

Система должна не просто сказать, что рынок может сделать. Она сохраняет состояние рынка в момент прогноза, то, что предсказала модель, к какому геному это состояние относилось — а затем то, что произошло на самом деле.

Через месяцы или годы можно открыть любую свечу и восстановить:

```
Состояние рынка
+ Features
+ Genome
+ Prediction
+ Model version
+ Actual result
```

### Возможности

**Данные и хранилище**
- Асинхронная загрузка OHLCV с Binance REST (`aiohttp`)
- Локальное хранилище SQLite (WAL), мультимонеты и мультитаймфреймы
- Автосинхронизация с настраиваемым интервалом
- Автоматическое нахождение и догрузка пропусков (окно `Coverage`)
- Прокси: DIRECT / HTTP / HTTPS / SOCKS4 / SOCKS5
- Network Monitor: latency, throughput, requests / errors / retries, обработка 429
- Auto-reconnect с exponential backoff

**Признаки (12 на свечу)**
- `return_1`, `return_5`, `return_20`
- `rsi`, `ema20`, `ema50`
- `volatility`, `volume_ratio`
- `high_low_range`, `body_size`, `upper_wick`, `lower_wick`

**Машинное обучение**
- LightGBM classifier (`UP` / `FLAT` / `DOWN`) и regressor (`expected_return`)
- Walk-forward валидация через `TimeSeriesSplit` — без утечки из будущего
- Мультигоризонт: 5m / 15m / 1h / 4h / 24h
- Изотоническая калибровка вероятностей
- Feature importance и SHAP-объяснения
- Мультитаймфрейм-консенсус (5m + 15m + 1h)
- Автоматическое переобучение при накоплении новых данных

**История прогнозов**
- Каждый прогноз сохраняется в SQLite
- Жизненный цикл `PENDING → RESOLVED`
- Автоматический resolver после наступления горизонта
- Реальная статистика: hit-rate, MAE, калибровочные бакеты
- Исторические стрелки прогнозов на графике (зелёная — угадал, красная — ошибся, серая — pending)

**Genome Engine**
- Косинусная кластеризация нормализованных feature-векторов
- `Genesis #0` — первый эталон, не заменяется
- Instances — все последующие обнаружения семейства
- Автоматическое создание нового Genome при низкой similarity
- Дерево Genome UI: `CCSMonitor → Монета → Genome → Genesis → Instances`
- Fingerprint (SHA1) для каждого instance

**Симуляция**
- Симуляция капитала по резолвнутым прогнозам
- Stop-loss / take-profit
- Position sizing: fixed / percent / Kelly
- Визуализация equity curve
- Таблица сделок с PnL и причиной выхода (SL / TP / горизонт)

**UI (Tkinter + Matplotlib)**
- Тёмная тема, flow-тулбар
- Свечной график + EMA20/EMA50 + панель RSI
- Зум под курсором, панорамирование, кроссхэйр, hover-тултип
- Стрелки прогнозов на обеих панелях
- Дерево Genome, Node Inspector
- Окна Predictions, Logs, Database
- Pipeline, Coverage, Storage Monitor, ML Insights, Consensus
- Tray-иконка (pystray), горячие клавиши, системные алерты

**Экспорт**
- CSV, XLSX, JSON
- Таблицы: candles, features, predictions, models, genomes, genome_instances, parse_runs, history_requests, simulations, simulation_trades

### Стек

```
Python 3.10+
├── SQLite3 (WAL)
├── aiohttp (Binance REST)
├── LightGBM (classifier + regressor)
├── scikit-learn (walk-forward, калибровка)
├── SHAP (объяснения)
├── NumPy, pandas
├── Tkinter + Matplotlib (UI)
├── pystray + Pillow (tray)
└── openpyxl (XLSX)
```

PyTorch не используется — MVP работает с табличными признаками и LightGBM.

### Установка

```bash
git clone https://github.com/CCSMonitor
cd CCSMonitor
python -m pip install -r requirements.txt
python app.py
```

### Требования

- Python 3.10+
- Доступ к Binance API (зависит от региона)
- ~50 МБ на БД, модели и логи изначально (растёт с историей)

### Использование

**Базовый сценарий**

1. Запустите `app.py`.
2. Выберите актив, таймфрейм и окно истории (`1y / 2y / 3y / 5y / Full`).
3. Нажмите `LOAD / SYNC`.
4. После загрузки данных нажмите `Train ML` (один раз).
5. Дальше `Auto ML` управляет переобучением и прогнозами автоматически.
6. `Predict` / `Predict All` — ручные прогнозы.
7. `Predictions` — реальная статистика hit-rate и MAE.
8. `Consensus` — сигнал по нескольким таймфреймам.
9. `Simulate` — прогон стратегии со SL/TP и position sizing.

**Горячие клавиши**

| Клавиша | Действие |
|---|---|
| `Ctrl+L` | Загрузить / синхронизировать |
| `Ctrl+R` | Sync now |
| `Ctrl+T` | Train ML |
| `Ctrl+P` | Predict |
| `Ctrl+Shift+P` | Predict по всем горизонтам |
| `Ctrl+Shift+C` | Consensus |
| `Ctrl+G` | Genomes |
| `Ctrl+I` | ML Insights |
| `Ctrl+S` | Simulate |
| `Ctrl+E` | Export |
| `Ctrl+0` | Сбросить зум |
| `+` / `−` | Зум |
| `F1`–`F6` | Network / Proxy / Pipeline / Coverage / Database / Logs |
| `Esc` | Закрыть активное окно |

### Конфигурация

Ключевые параметры в `config.py`:

```python
ML_HORIZON_SECONDS = 3600          # горизонт прогноза по умолчанию
ML_FLAT_THRESHOLD = 0.003          # ±0.3% — граница UP/DOWN и FLAT
ML_HORIZONS = [("5m", 300), ("15m", 900), ("1h", 3600), ("4h", 14400), ("24h", 86400)]

AUTO_TRAIN_MIN_INTERVAL_SEC = 6 * 3600     # минимум 6 часов между тренировками
AUTO_TRAIN_MIN_NEW_CANDLES = 500           # минимум новых свечей

GENOME_SIMILARITY_THRESHOLD = 0.90         # порог косинусной близости
CONSENSUS_TIMEFRAMES = ["5m", "15m", "1h"]

SIMULATION_STOP_LOSS_PCT = 0.02
SIMULATION_TAKE_PROFIT_PCT = 0.04
SIMULATION_SIZING_MODE = "fixed"           # fixed | percent | kelly
```

### Принципы против утечки данных

- X[i] использует только прошлое и текущее.
- y[i] = `close[i+h]/close[i] − 1` — только как target, никогда в X.
- Обучение на первых `n − h` свечах (target никогда не выходит за пределы ряда).
- Walk-forward через `TimeSeriesSplit`, без перемешивания.
- Прогноз делается на последней свече с полным feature-вектором.
- Нормализация fit только на train, transform на валидации.

### Что система НЕ делает

- Не исполняет сделки (это не бот).
- Нет WebSocket (пока REST polling с настраиваемым интервалом).
- Нет Parquet (всё в SQLite).
- Не даёт сигналов «купи/продай» — только статистические вероятности и измерение точности.
- Не гарантирует точность или прибыльность.

### Структура проекта

```
CCSMonitor/
├── app.py                # точка входа
├── config.py             # все константы
├── database.py           # SQLite + схема + методы
├── market.py             # асинхронный клиент Binance
├── async_engine.py       # event loop, sync, ML, genome, resolver
├── features.py           # 12 признаков
├── ml.py                 # LightGBM, walk-forward, калибровка, SHAP
├── genome.py             # кластеризация, fingerprint, имена геномов
├── network.py            # NetworkStats
├── storage.py            # размеры на диске
├── simulation.py         # капитал / SL / TP / sizing
├── export.py             # CSV / XLSX / JSON
├── tray.py               # tray-иконка (pystray)
├── alerts.py             # уведомления
├── ui.py                 # Tkinter + Matplotlib UI
├── requirements.txt
├── data/                 # опционально Parquet (пока не используется)
├── models/               # обученные бустеры
├── logs/                 # опционально логи в файл
└── exports/              # экспортированные таблицы
```

### Roadmap

**Готово**
- [x] Data Core (REST + SQLite + WAL)
- [x] Графики (свечи + EMA + RSI)
- [x] Feature Engine (12 признаков)
- [x] ML (LightGBM + walk-forward)
- [x] Prediction History + Resolver
- [x] Genome Engine + Genome UI
- [x] Полная загрузка истории (1y–5y–Full)
- [x] Network Monitor, Proxy UI, Auto-reconnect, 429
- [x] Pipeline UI, Coverage, Database Health, Storage Monitor
- [x] Feature importance, SHAP, калибровка
- [x] Мультигоризонт, мультитаймфрейм-консенсус
- [x] Equity curve, SL/TP, position sizing
- [x] Логи, экспорт CSV/XLSX/JSON, симуляция
- [x] Горячие клавиши, tray-иконка, алерты

### Лицензия

GNU GPL v3.

### Дисклеймер

Это программное обеспечение предоставляется **исключительно в исследовательских и образовательных целях**. Это не финансовая рекомендация. Криптовалютные рынки крайне волатильны. Любое использование в реальной торговле — на ваш риск. Авторы не несут ответственности за финансовые потери.

---

**Keywords:** cryptocurrency, machine-learning, lightgbm, time-series, walk-forward, backtesting, python, tkinter, sqlite, binance, technical-analysis, shap, research, open-source, genome
```