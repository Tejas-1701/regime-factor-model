from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(page_title="Regime-Aware Factor Model", layout="wide")


@st.cache_data
def load_data():
    master = pd.read_csv(DATA_DIR / "master_with_regimes.csv", index_col=0, parse_dates=True)
    portfolio = pd.read_csv(DATA_DIR / "portfolio_results.csv", index_col=0, parse_dates=True)
    return master, portfolio


def performance(monthly_returns):
    months = len(monthly_returns)
    annual_return = (1 + monthly_returns).prod() ** (12 / months) - 1
    sharpe = monthly_returns.mean() / monthly_returns.std() * np.sqrt(12)
    growth = (1 + monthly_returns).cumprod()
    max_drawdown = (growth / growth.cummax() - 1).min()
    return annual_return, sharpe, max_drawdown


master, portfolio = load_data()

gross_return, gross_sharpe, gross_drawdown = performance(portfolio["Portfolio_Return"])
net_return, net_sharpe, net_drawdown = performance(portfolio["Portfolio_Return_Net"])
market_return, market_sharpe, market_drawdown = performance(portfolio["Market_Return"])

start_year = portfolio.index.min().year
end_year = portfolio.index.max().year

st.title("Beyond Beta: Regime-Aware ML Factor Model")
st.markdown(f"S&P 500 · Backtest {start_year}–{end_year} · HMM Regime Detection · XGBoost · 501 Stocks")
st.divider()

regime_emoji = {"Bull": "🐂", "Bear": "🐻", "Volatile": "📉"}
current_regime = master["Regime_Label"].iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Regime", f"{regime_emoji.get(current_regime, '')} {current_regime}")
col2.metric("Annual Return", f"{gross_return:.2%}", f"{gross_return - market_return:+.2%} vs S&P 500")
col3.metric("Sharpe Ratio", f"{gross_sharpe:.2f}", f"{gross_sharpe - market_sharpe:+.2f} vs S&P 500")
col4.metric("Excess Return", f"{gross_return - market_return:+.2%}", "per year, before costs")
col5.metric("Max Drawdown", f"{gross_drawdown:.2%}", f"S&P 500: {market_drawdown:.2%}", delta_color="off")

st.divider()

regime_colors = {"Bull": "green", "Bear": "red", "Volatile": "orange"}

st.subheader("Cumulative Returns — Portfolio vs S&P 500")
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(portfolio.index, portfolio["Cumulative_Portfolio"], color="blue", linewidth=1.8, label="Regime-Aware Portfolio")
ax.plot(portfolio.index, portfolio["Cumulative_Market"], color="gray", linewidth=1.5, linestyle="--", label="S&P 500")
ax.plot(portfolio.index, portfolio["Cumulative_Portfolio_Net"], color="cornflowerblue", linewidth=1.2, linestyle=":", label="Portfolio (net of costs)")
for regime, color in regime_colors.items():
    for date in portfolio.index[portfolio["Regime"] == regime]:
        ax.axvspan(date, date + pd.DateOffset(months=1), alpha=0.08, color=color)
ax.set_ylabel("Growth of $1")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Regime Timeline")
    fig2, ax2 = plt.subplots(figsize=(7, 2))
    for regime, color in regime_colors.items():
        in_regime = master["Regime_Label"] == regime
        ax2.scatter(master.index[in_regime], [1] * in_regime.sum(), color=color, s=40, label=regime, marker="s")
    ax2.set_yticks([])
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)

    st.subheader("Backtest Summary")
    summary = pd.DataFrame({
        "Metric": ["Annual Return", "Sharpe Ratio", "Max Drawdown"],
        "Portfolio": [f"{gross_return:.2%}", f"{gross_sharpe:.2f}", f"{gross_drawdown:.2%}"],
        "Net of Costs": [f"{net_return:.2%}", f"{net_sharpe:.2f}", f"{net_drawdown:.2%}"],
        "S&P 500": [f"{market_return:.2%}", f"{market_sharpe:.2f}", f"{market_drawdown:.2%}"],
    })
    st.dataframe(summary, hide_index=True)

with col_right:
    st.subheader("Feature Importance by Regime")
    st.image(str(DATA_DIR / "feature_importance.png"))

st.divider()

st.subheader("Current Macro Environment")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("VIX", f"{master['VIX'].iloc[-1]:.2f}", "Fear Index", delta_color="off")
col_b.metric("Yield Spread", f"{master['YieldSpread'].iloc[-1]:.2f}%", "10Y - 2Y", delta_color="off")
col_c.metric("GPR Global", f"{master['GPR_Global'].iloc[-1]:.1f}", "Geopolitical Risk", delta_color="off")
col_d.metric("Oil Price", f"${master['Oil_Price'].iloc[-1]:.2f}")

st.divider()

st.subheader("Last 12 Months — Macro Data")
display_columns = ["Mkt-RF", "VIX", "YieldSpread", "GPR_Global", "Oil_Price", "Gold_Price", "Regime_Label"]
st.dataframe(master.tail(12)[display_columns].round(4))
