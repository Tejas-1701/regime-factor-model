import sys
sys.path.insert(0, r'E:\regime_factor_model\libs')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Regime-Aware Factor Model", layout="wide")

# Load data
master = pd.read_csv(r'E:\regime_factor_model\data\master_with_regimes.csv',
                     index_col=0, parse_dates=True)
portfolio = pd.read_csv(r'E:\regime_factor_model\data\portfolio_results.csv',
                        index_col=0, parse_dates=True)

# Header
st.title("Beyond Beta: Regime-Aware ML Factor Model")
st.markdown("S&P 500 · 2004–2024 · HMM Regime Detection · XGBoost · 501 Stocks")
st.divider()

# Current regime
current_regime = master['Regime_Label'].iloc[-1]
current_vix = master['VIX'].iloc[-1]
current_gpr = master['GPR_Global'].iloc[-1]
current_yield = master['YieldSpread'].iloc[-1]

regime_emoji = {'Bull': '🐂', 'Bear': '🐻', 'Volatile': '📉'}

# Metrics row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Regime", f"{regime_emoji[current_regime]} {current_regime}")
col2.metric("Annual Return", "19.40%", "+6.74% vs S&P 500")
col3.metric("Sharpe Ratio", "1.17", "+0.30 vs market")
col4.metric("Alpha", "6.74%", "per year gross")
col5.metric("Max Drawdown", "-25.88%", "vs -25.26% market")

st.divider()

# Cumulative returns
st.subheader("Cumulative Returns — Portfolio vs S&P 500")
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(portfolio.index, portfolio['Cumulative_Portfolio'],
        color='blue', linewidth=1.8, label='Regime-Aware Portfolio')
ax.plot(portfolio.index, portfolio['Cumulative_Market'],
        color='gray', linewidth=1.5, linestyle='--', label='S&P 500')
ax.plot(portfolio.index, portfolio['Cumulative_Portfolio_Net'],
        color='cornflowerblue', linewidth=1.2, linestyle=':', label='Portfolio (net of costs)')
colors = {'Bull': 'green', 'Bear': 'red', 'Volatile': 'orange'}
for regime, color in colors.items():
    mask = portfolio['Regime'] == regime
    for date in portfolio.index[mask]:
        ax.axvspan(date, date + pd.DateOffset(months=1),
                   alpha=0.08, color=color)
ax.set_ylabel('Growth of $1')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.divider()

# Two columns
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Regime Timeline")
    fig2, ax2 = plt.subplots(figsize=(7, 2))
    for regime, color in colors.items():
        mask = master['Regime_Label'] == regime
        ax2.scatter(master.index[mask], [1]*mask.sum(),
                    color=color, s=40, label=regime, marker='s')
    ax2.set_yticks([])
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)

    st.subheader("Backtest Summary")
    summary = pd.DataFrame({
        'Metric': ['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Alpha'],
        'Portfolio': ['19.40%', '1.17', '-25.88%', '+6.74%'],
        'Net of Costs': ['18.00%', '1.09', '-26.05%', '+5.34%'],
        'S&P 500': ['12.66%', '0.87', '-25.26%', '—']
    })
    st.dataframe(summary, hide_index=True)

with col_right:
    st.subheader("Feature Importance by Regime")
    st.image(r'E:\regime_factor_model\data\feature_importance.png')

st.divider()

# Macro indicators
st.subheader("Current Macro Environment")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("VIX", f"{current_vix:.2f}", "Fear Index")
col_b.metric("Yield Spread", f"{current_yield:.2f}%", "10Y - 2Y")
col_c.metric("GPR Global", f"{current_gpr:.1f}", "Geopolitical Risk")
col_d.metric("Oil Price", f"${master['Oil_Price'].iloc[-1]:.2f}")

st.divider()

st.subheader("Last 12 Months — Macro Data")
display_cols = ['Mkt-RF', 'VIX', 'YieldSpread', 'GPR_Global',
                'Oil_Price', 'Gold_Price', 'Regime_Label']
st.dataframe(master.tail(12)[display_cols].round(4))