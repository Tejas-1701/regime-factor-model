# Beyond Beta: Regime-Aware ML Factor Model

Most stock prediction models treat the market as if it always behaves the same way. It doesn't. A model that works well during a calm bull run will fall apart during a crash — and that's precisely when you need it most.

This project takes a different approach. Instead of fitting one model to all market conditions, it first figures out what kind of market environment we're in, then applies a model specifically trained for that environment.

Built on 22 years of S&P 500 data (2004–2026) across 501 stocks.

---

## Results

| Metric | Our Portfolio | S&P 500 |
|---|---|---|
| Annual Return | 18.53% | 12.88% |
| Sharpe Ratio | 1.10 | 0.90 |
| Max Drawdown | -28.64% | -25.26% |
| Alpha | +5.65% per year | — |

---

## How It Works

The pipeline has five stages:

**1. Data Collection**
Stock prices for all 501 S&P 500 companies going back to 2004, combined with Fama-French factors, macro indicators from FRED (VIX, yield curve, Fed Funds rate), commodity prices (oil and gold), and the Geopolitical Risk Index from Caldara and Iacoviello (2022). Everything is cleaned and aligned into a single monthly master dataset.

**2. Statistical Analysis**
Before building any model, we tested which factors actually matter. Market returns (Mkt-RF) and profitability (RMW) are statistically significant. Size (SMB), value (HML), and investment (CMA) have weakened considerably since 2010 — a finding consistent with recent academic literature.

**3. Regime Detection**
A Gaussian Hidden Markov Model looks at market returns, VIX, yield spread, geopolitical risk, and oil prices to classify each month into one of three regimes:

- Bull — calm, rising market. VIX below 15, positive yield spread.
- Bear — falling or stagnant market. VIX above 19, often inverted yield curve.
- Volatile — elevated uncertainty. VIX above 23, high geopolitical risk.

**4. Regime-Aware Portfolio Strategy**
Each regime triggers a different strategy:
- Bull → buy the top 20 stocks by recent momentum
- Bear → buy the 20 lowest-volatility stocks (defensive positioning)
- Volatile → buy the top 20 stocks by short-term momentum

**5. Feature Importance**
The model learns that different signals matter in different regimes. In bull markets, the yield spread is the dominant signal. In bear markets, gold price takes over — investors are fleeing to safety. In volatile periods, short-term momentum is what matters most. This is exactly what financial theory would predict.

---

## Current Market Status (April 2026)

The model has been detecting a Bear regime since January 2026. VIX climbed from 14.95 in December 2025 to 25.25 by March 2026, driven by geopolitical tensions and macro uncertainty. The strategy automatically shifted to low-volatility defensive stocks in response.

---

## Project Structure
regime-factor-model/
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   └── 02_regime_detection.ipynb
├── dashboard/
│   └── app.py
├── data/
└── README.md

---

## Data Sources

| Source | What it provides |
|---|---|
| Yahoo Finance | Daily prices for 501 S&P 500 stocks |
| Ken French Data Library | Fama-French 5 factors |
| FRED | VIX, yield curve, Fed Funds rate, unemployment |
| Caldara and Iacoviello 2022 | Geopolitical Risk Index |
| Yahoo Finance | WTI crude oil and gold prices |

---

## How to Run

Clone the repo and install dependencies:

pip install -r requirements.txt

Run the notebooks in order starting from 01_data_pipeline.ipynb.

To launch the dashboard locally:

streamlit run dashboard/app.py

---

## Stack

Python 3.10 · PySpark · XGBoost · hmmlearn · Streamlit · scikit-learn · yfinance · pandas-datareader · Plotly