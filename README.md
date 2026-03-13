# ✦ Tira Beauty · Brand Intelligence Dashboard

> **MBA Project** · Data Analytics & Decision Making  
> Research Question: *Is Discounting Killing Premium Brand Perception?*

---

## Overview

A Streamlit-powered analytics dashboard investigating how Tira Beauty's discount strategy impacts brand perception, customer loyalty, and revenue sustainability. Built using synthetic data and four ML models.

## Analytics Modules

| Module | Technique | Business Question |
|---|---|---|
| 🎯 Classification | Random Forest + Logistic Regression | Which customers are becoming discount-dependent? |
| 👥 Clustering | K-Means (5 clusters) | What customer archetypes exist in Tira's base? |
| 🔗 Association Rules | Apriori / FP-Growth | What purchase behaviors co-occur with discounting? |
| 📈 Regression | Linear Regression | How does discounting forecast revenue & NPS? |

## Visualizations

- **Treemap** — Revenue by Brand Tier × Category
- **Sankey Diagram** — Customer journey: Channel → Loyalty Outcome
- **Waterfall Chart** — Revenue impact of discount leakage
- **Radar Chart** — Persona profile comparison
- **Scatter Plots** — Discount exposure vs NPS / Basket Size
- **Heatmap** — Category co-purchase matrix
- **Forecast Line Chart** — 12-month scenario analysis

## Setup & Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/tira-brand-dashboard.git
cd tira-brand-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

## File Structure

```
tira-brand-dashboard/
├── app.py               # Main Streamlit dashboard
├── data_generator.py    # Synthetic dataset generation (2,000 customers)
├── analytics.py         # ML models: Classification, Clustering, ARM, Regression
├── requirements.txt     # Python dependencies
└── README.md
```

## Key Findings

- Every **10% increase** in discount exposure = **₹320 drop** in avg basket size
- Crossing the **25% discount threshold** triggers measurable NPS decline
- **Prestige Loyalists** (23% of base) generate 3.2× more revenue than Deal Hunters
- Flash sale buyers show **61% lower** 30-day repurchase rates
- Controlled discounting projects **+20% revenue** vs aggressive strategy's **-30%** over 12 months

---

*Dataset: 2,000 synthetic customers · 5,000 synthetic transactions · Dark Apple-inspired UI*
