# S&P 500 Next-Day Direction: Can Simple Technical Indicators Beat a Naive Baseline?

A machine learning project that tests whether a Random Forest, trained on simple technical indicators, can predict the next-day direction of the S&P 500 better than a naive "always up" baseline.

**Main finding:** it cannot. On this data, the model does not beat the baseline in accuracy, precision, Sharpe ratio or drawdown. The project is built to reach that conclusion rigorously, without data leakage and with realistic evaluation.

## Question

Can a Random Forest using price-based indicators predict whether the S&P 500 closes higher tomorrow, and does trading on those predictions beat buy & hold?

## Data

- S&P 500 index (`^GSPC`), daily prices from 2014-01-01 to 2024-12-31, downloaded with `yfinance`.
- 2,566 usable trading days after removing the rows lost to rolling windows (2,052 for training and 514 for testing).

## Method

**Target.** 1 if tomorrow's close is higher than today's close, 0 otherwise.

**Features.** All features are scale-free (returns and ratios rather than raw prices), because a Random Forest cannot extrapolate beyond the price levels seen in training:

| Feature | Description |
|---|---|
| `Daily_Return` | Percentage change of the close |
| `RSI` | 14-day Relative Strength Index |
| `Volatility` | Standard deviation of daily returns over 10 days |
| `Volume_Change` | Percentage change of volume |
| `HL_Range` | (High - Low) / Close |
| `Close_vs_MA50` | Close / 50-day moving average - 1 |
| `Close_vs_MA200` | Close / 200-day moving average - 1 |

**Validation.**
- Chronological 80/20 split (no shuffling, to avoid look-ahead bias).
- Walk-forward validation with `TimeSeriesSplit` (5 folds), always training on the past and testing on the following block.

**Model.** `RandomForestClassifier(n_estimators=200, min_samples_split=100, random_state=1)`.

**Baseline.** Always predicting "up". Since the S&P 500 rises on most days, this baseline is hard to beat and is the right reference for accuracy and precision.

**Backtest.** Long/cash strategy: hold the index the day after an "up" prediction, stay in cash otherwise. The prediction made on day *t* is applied to the return of day *t+1*. Transaction costs of 0.1% are charged at every position change. Sharpe ratios are annualised with 252 trading days and a risk-free rate of 0.

## Results (test period, roughly the last two years of the sample)

| Metric | Model | Baseline / Buy & hold |
|---|---|---|
| Accuracy | 52.92% | 55.45% (always up) |
| Precision (on "up") | 55.26% | 55.45% (always up) |
| Walk-forward accuracy (mean of 5 folds) | 51.85% | 54.47% (always up) |
| Sharpe ratio, no costs | 1.17 | 1.60 |
| Sharpe ratio, after costs | 0.54 | 1.60 |
| Max drawdown, no costs | -11.18% | -10.28% |
| Max drawdown, after costs | -11.71% | -10.28% |
| Growth of 1 unit, no costs | 1.32 | 1.50 |
| Growth of 1 unit, after costs | 1.13 | 1.50 |
| Position changes | 159 | 0 |

## Conclusion

- The model is less accurate than always predicting "up", and its precision is practically the same as the baseline, so its "up" signals carry no extra information.
- Walk-forward validation gives the same picture, so the result does not depend on one particular split.
- The strategy has a lower Sharpe ratio than buy & hold, and costs reduce it further (1.17 to 0.54). It did not reduce the maximum drawdown either.
- Next-day direction from simple price indicators is very close to noise, and the strong upward drift of the index makes staying in cash costly.

## Limitations

- One index and one time period.
- A single set of hyperparameters, not tuned.
- Simplified cost model (0.1% per position change), with no slippage or taxes.
- The test period is about two years, so the estimates have wide uncertainty.

## Possible next steps

- Add macroeconomic variables and test other models.
- Predict returns over several days instead of the next day.
- Tune hyperparameters inside a walk-forward scheme.
- Evaluate on other assets and periods.

## Repository

- `s-p-500-price-prediction.ipynb`: the full analysis (run top to bottom).
- `LICENSE`: Apache-2.0.

## How to run

Open the notebook in Google Colab and run all cells (`Runtime > Run all`). Requirements: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `yfinance`.
---

## 👤 Author

**Augusto Bernardo Novais**
[LinkedIn](https://www.linkedin.com/in/augusto-bernardo-novais-2b77b2409)
