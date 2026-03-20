# PrimeTrade.ai — Trader Behavior Insights Engine

A data science project to analyze how Bitcoin market sentiment (Fear vs Greed) influences trader behavior and trade outcomes, built as an internship assignment aligned with real trading intelligence workflows.

## About Primetrade.ai
Primetrade.ai is a niche AI and blockchain venture studio, where we help multiple product startups grow in cutting-edge fields of AI and blockchain.

---

## Problem Statement
Crypto markets are highly sentiment-driven, but most trading analysis ignores how trader behavior shifts across emotional regimes.  
This project studies whether sentiment state (`Fear` / `Greed`) is linked to profitability, leverage risk, position behavior, and strategy edge.

---

## Dataset Description
### 1) Hyperliquid Historical Trader Data
Trade-level fields include:
- account
- coin/symbol
- execution price
- size (tokens / USD)
- side / direction
- timestamps
- closed PnL
- leverage (if available)
- fees and execution metadata

### 2) Bitcoin Market Sentiment Data
Daily sentiment labels:
- Date
- Classification (`Fear` / `Greed`)

---

## Approach
The project is implemented in a phase-wise, production-style pipeline:

1. **Data preprocessing**
   - Standardized column names
   - Datetime parsing and date-level alignment
   - Missing value handling
   - Feature engineering (`profit`, `is_profit`, `trade_size_usd`)

2. **Data merge**
   - Left join of trader data with daily sentiment on aligned `date`
   - Merge quality checks (null sentiment coverage)

3. **EDA and behavioral analytics**
   - Profit and win-rate by sentiment
   - Leverage behavior by sentiment
   - Buy/sell bias by sentiment
   - Profit distribution analysis

4. **Advanced insights**
   - Top-trader identification and cohort comparison
   - Regime behavior (`Fear` vs `Greed`) among top traders
   - Contrarian vs trend-following performance tests

5. **Modeling + segmentation**
   - Baseline profitability prediction model
   - Trader clustering into profile groups (e.g., risk-takers, conservative, consistent)

6. **Explainable AI (SHAP)**
   - Global feature influence ranking
   - Local explanations for individual trade predictions
   - Trading-desk style interpretation of model behavior

---

## Key Insights (Highlights)
- **Sentiment matters:** trade profitability and win rate change across `Fear` and `Greed` regimes.
- **Alpha concentration exists:** a small group of top accounts contributes a disproportionate share of total PnL.
- **Top traders are adaptive:** stronger accounts change exposure and behavior by regime instead of using fixed execution patterns.
- **Risk behavior is regime-sensitive:** leverage and directional bias shift with sentiment state.
- **Explainability confirms intuition:** SHAP shows that leverage, position size, execution context, and sentiment features are key drivers of profit/loss predictions.

---

## Sample Results
### A) EDA Outcomes
- Average profit and win-rate vary by sentiment class
- Distinct buy/sell composition in Fear vs Greed
- Different profit dispersion across sentiment regimes

### B) Trader Intelligence Outcomes
- Ranked top traders by total profit
- Clear behavioral gap between top cohort and average cohort
- Measurable difference between contrarian and trend-following styles

### C) Model + XAI Outcomes
- Baseline model provides meaningful predictive signal for `is_profit`
- SHAP global plot identifies top impact features
- Local SHAP explanations show why specific trades were predicted as profit/loss

> SHAP artifacts are generated in the outputs folder after running the XAI script.

---

## Project Structure
- [preprocess_phase2.py](preprocess_phase2.py) — data cleaning, feature engineering, merge logic
- [eda_phase4.py](eda_phase4.py) — exploratory analysis with plots + interpretations
- [insights_phase5.py](insights_phase5.py) — top-trader and contrarian insights
- [phase6_modeling.py](phase6_modeling.py) — baseline prediction + trader clustering
- [phase7_report.md](phase7_report.md) — professional summary report
- [phase8_xai_shap.py](phase8_xai_shap.py) — SHAP explainable AI pipeline
- [run_all.py](run_all.py) — run the full project in one command

---

## Quick Run (One Command)
1. Keep both dataset files in project root:
   - [historical_data (1).csv](historical_data%20(1).csv)
   - [fear_greed_index.csv](fear_greed_index.csv)
2. Install dependencies:
   - `pip install pandas numpy matplotlib scikit-learn shap`
3. Run everything:
   - `python run_all.py`

---

## Conclusion
This project demonstrates how to convert raw trade logs and market sentiment into practical trading intelligence.  
The core value is not only prediction accuracy, but **decision-quality improvement**: regime-aware risk management, profile-based capital allocation, and explainable signals that a trading firm can trust and operationalize.

In short: this framework helps a trading desk move from retrospective analysis to actionable, sentiment-aware strategy execution.
# primetradeAI-assignment
