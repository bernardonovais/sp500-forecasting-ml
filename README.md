# sp500-forecasting-ml
# 📈 S&P 500 Price Movement Prediction using Machine Learning

An end-to-end Machine Learning project that predicts the **daily price movement direction** of the S&P 500 index using a Random Forest Classifier and technical indicators, including a backtesting simulation to evaluate the strategy's performance.

---

## 🎯 Objective

Predict whether the S&P 500 closing price will go **up (1)** or **down (0)** the next day, using 10 years of historical data (2014–2024).

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.12` | Core language |
| `yfinance` | Historical market data retrieval |
| `pandas` & `numpy` | Data manipulation |
| `scikit-learn` | Random Forest model & evaluation |
| `matplotlib` & `seaborn` | Data visualization |

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `MA50` | 50-day Moving Average |
| `MA200` | 200-day Moving Average |
| `RSI` | Relative Strength Index (14 periods) |
| `Daily_Return` | Daily percentage price change |
| `Volatility` | 10-day rolling standard deviation |
| `Volume_Change` | Percentage change in trading volume |
| `HL_Range` | High-Low daily price spread |

---

## 🧠 Methodology

1. **Data Collection** — 10 years of S&P 500 data via `yfinance`
2. **Feature Engineering** — Technical indicators calculated from raw OHLCV data
3. **Target Labeling** — Binary classification: `1` if next day's price > today's, `0` otherwise
4. **Chronological Split** — 80% training / 20% testing (no data leakage)
5. **Model** — `RandomForestClassifier` with 200 estimators
6. **Backtesting** — Simulated trading strategy vs. Buy & Hold benchmark

---

## 📈 Results

- **Model Accuracy:** ~55%
- **Precision on 'UP' signals:** ~55%
- **Key insight:** In quantitative finance, a consistent 5% edge over random chance is considered highly significant

> While the strategy underperformed the Buy & Hold benchmark in cumulative returns, the model demonstrated a meaningful precision edge on bullish signals — avoiding some of the downside risk in exchange for missing recovery rallies.

---

## 🚀 Future Improvements

- Sentiment Analysis (NLP) from financial news headlines
- Hyperparameter tuning with Optuna
- Macroeconomic indicators (interest rates, inflation)
- Ensemble models (XGBoost, LightGBM)

---

## 📁 Project Structure

```
sp500-forecasting-ml/
│
└── s-p-500-price-prediction.ipynb   # Full notebook: EDA, feature engineering, model, backtesting
```

---

## 👤 Author

**Augusto Bernardo Novais**
[LinkedIn](https://www.linkedin.com/in/augusto-bernardo-novais-2b77b2409)
