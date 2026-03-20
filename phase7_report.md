# Trader Behavior vs Bitcoin Sentiment — Professional Summary Report

## Executive Summary
This project analyzed Hyperliquid trade-level performance against Bitcoin market sentiment regimes (`Fear` vs `Greed`) to identify repeatable behavioral edges. The analysis combined robust preprocessing, sentiment alignment, exploratory diagnostics, top-trader profiling, contrarian tests, and baseline modeling/clustering. The result is a practical decision framework for risk allocation, execution style, and regime-aware strategy design.

## Top 5 Insights

1. **Sentiment materially affects trade outcomes.**  
   Profitability and win rate are not constant across regimes; trader performance shifts between `Fear` and `Greed`, confirming that regime context should be treated as a core trading variable.

2. **Alpha is concentrated in a small trader cohort.**  
   Top traders contribute a disproportionate share of aggregate PnL, with stronger per-trade edge and better consistency than the average cohort.

3. **Top traders adapt behavior by regime.**  
   The best accounts adjust exposure (size/leverage/direction) depending on `Fear` vs `Greed`, rather than trading with a fixed playbook.

4. **Directional behavior changes with sentiment.**  
   Buy/sell composition is regime-dependent, indicating that trader positioning and conviction are sentiment-sensitive.

5. **A simple predictive baseline already captures useful signal.**  
   Even an interpretable baseline model can partially predict profitable trades using sentiment + risk/execution features, proving that outcome signal is real and exploitable.

## Recommended Trading Strategies (2–3)

1. **Regime-Aware Position Sizing**  
   Increase/decrease position size dynamically based on current sentiment regime and account-specific historical edge under that regime.

2. **Sentiment-Conditioned Risk Limits**  
   Apply tighter leverage caps and stricter entry thresholds in weak regimes; relax selectively for profiles that retain edge during those conditions.

3. **Style Rotation (Contrarian vs Trend-Following)**  
   Allocate capital to contrarian or trend-following execution only when that style shows superior recent regime-adjusted expectancy.

## Business Impact for a Trading Firm

1. **Higher risk-adjusted returns:** capital is routed to the right strategy-account-regime combinations instead of static allocation.
2. **Lower drawdown risk:** sentiment-triggered guardrails reduce overexposure during unfavorable market states.
3. **Faster decision quality:** a repeatable analytics stack turns raw trade logs into actionable, daily risk and execution signals.
4. **Scalable trader evaluation:** objective profile segmentation supports better onboarding, capital assignment, and performance governance.

## Final Position
This framework is immediately useful for production-like decision support: it links market mood to trader behavior, quantifies who performs best and when, and translates that into concrete allocation and risk controls.
