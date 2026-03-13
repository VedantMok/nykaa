import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Tira Beauty · Brand Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path("tira_customers.csv")
GOLD = "#d4a843"
BLUE = "#2997ff"
GREEN = "#30d158"
ROSE = "#e8657a"
PURPLE = "#bf5af2"
ORANGE = "#ff9f0a"
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#d2d2d7", size=12),
        title=dict(font=dict(color="#f5f5f7", size=16), x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(17,17,17,0.75)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
        colorway=[GOLD, BLUE, GREEN, ROSE, PURPLE, ORANGE],
        margin=dict(l=10, r=10, t=45, b=10),
    )
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0b0b0d 0%, #121216 100%);
            color: #f5f5f7;
        }
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .section-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 18px;
            margin-bottom: 16px;
        }
        h1, h2, h3 { color: #f5f5f7; }
        .stTabs [data-baseweb="tab"] { font-size: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "city_tier", "income_bracket", "acquisition_channel",
    "num_visits", "avg_basket_size", "discount_pct", "category", "brand_tier",
    "flash_sale_response", "flash_sale_frequency", "nps_score", "basket_trend",
    "willingness_full_price", "repurchase_intent", "months_active", "premium_loyal"
]


def ensure_dataset(path: Path) -> None:
    if path.exists():
        return
    rng = np.random.default_rng(42)
    n = 2000
    customer_ids = [f"TIRA{str(i).zfill(5)}" for i in range(1, n + 1)]
    ages = np.clip(rng.normal(30, 8, n), 18, 60).astype(int)
    genders = rng.choice(["Female", "Male", "Non-binary"], size=n, p=[0.72, 0.23, 0.05])
    city_tiers = rng.choice(["Tier 1", "Tier 2", "Tier 3"], size=n, p=[0.55, 0.30, 0.15])
    income_brackets = rng.choice(["₹3L-6L", "₹6L-12L", "₹12L-25L", "₹25L+"], size=n, p=[0.15, 0.35, 0.35, 0.15])
    acquisition_channels = rng.choice(["App", "In-Store", "Influencer", "Paid Ad", "Word of Mouth"], size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10])
    num_visits = np.clip(rng.poisson(8, n), 1, 40)
    avg_basket_size = np.clip(rng.normal(3200, 1200, n), 500, 12000)
    discount_pct = rng.beta(2, 5, n) * 100
    discount_pct = np.where(city_tiers == "Tier 3", np.minimum(discount_pct * 1.4, 80), discount_pct)
    discount_pct = np.where(income_brackets == "₹3L-6L", np.minimum(discount_pct * 1.3, 80), discount_pct)
    categories = rng.choice(["Skincare", "Makeup", "Haircare", "Fragrance", "Luxury Tools", "Wellness"], size=n, p=[0.35, 0.28, 0.15, 0.12, 0.05, 0.05])
    brand_tier = rng.choice(["Masstige", "Premium", "Luxury"], size=n, p=[0.30, 0.45, 0.25])
    flash_sale_response = (discount_pct > 30).astype(int)
    flash_sale_frequency = np.clip(rng.poisson(3, n), 0, 15)
    nps_score = np.clip(8 - (discount_pct / 25) + rng.normal(0, 1, n), 1, 10)
    basket_trend = avg_basket_size * (1 - discount_pct / 200)
    willingness_full_price = (discount_pct < 25).astype(int)
    repurchase_intent = (nps_score > 7).astype(int)
    months_active = rng.integers(1, 36, n)
    score = (
        (discount_pct < 20).astype(int) * 2
        + (nps_score > 7).astype(int) * 2
        + (avg_basket_size > 3000).astype(int)
        + (brand_tier == "Luxury").astype(int) * 2
        + (brand_tier == "Premium").astype(int)
        + rng.normal(0, 0.5, n)
    )
    premium_loyal = (score > 4).astype(int)
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "city_tier": city_tiers,
        "income_bracket": income_brackets,
        "acquisition_channel": acquisition_channels,
        "num_visits": num_visits,
        "avg_basket_size": np.round(avg_basket_size, 0),
        "discount_pct": np.round(discount_pct, 1),
        "category": categories,
        "brand_tier": brand_tier,
        "flash_sale_response": flash_sale_response,
        "flash_sale_frequency": flash_sale_frequency,
        "nps_score": np.round(nps_score, 1),
        "basket_trend": np.round(basket_trend, 0),
        "willingness_full_price": willingness_full_price,
        "repurchase_intent": repurchase_intent,
        "months_active": months_active,
        "premium_loyal": premium_loyal,
    })
    df.to_csv(path, index=False)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {', '.join(missing)}")
        st.stop()


@st.cache_data
def load_data(uploaded_file=None):
    ensure_dataset(DATA_FILE)
    df = pd.read_csv(uploaded_file if uploaded_file is not None else DATA_FILE)
    validate_columns(df)
    df["estimated_revenue"] = df["avg_basket_size"] * df["num_visits"]
    df["loyalty_label"] = df["premium_loyal"].map({1: "Premium Loyal", 0: "Discount-Dependent"})
    df["discount_band"] = pd.cut(
        df["discount_pct"],
        bins=[-0.1, 10, 20, 30, 40, 100],
        labels=["0-10%", "10-20%", "20-30%", "30-40%", "40%+"],
    )
    return df


@st.cache_data
def run_classification(df):
    features = ["age", "num_visits", "avg_basket_size", "discount_pct", "flash_sale_frequency", "nps_score", "months_active"]
    X = df[features]
    y = df["premium_loyal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    lr = LogisticRegression(max_iter=1500, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return {
        "rf_accuracy": round(accuracy_score(y_test, rf_pred) * 100, 1),
        "lr_accuracy": round(accuracy_score(y_test, lr_pred) * 100, 1),
        "rf_auc": round(roc_auc_score(y_test, rf_proba) * 100, 1),
        "feature_importances": importances,
    }


@st.cache_data
def run_clustering(df):
    features = ["avg_basket_size", "discount_pct", "nps_score", "num_visits", "flash_sale_frequency"]
    X_scaled = StandardScaler().fit_transform(df[features])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    clustered = df.copy()
    clustered["cluster"] = labels

    cluster_stats = clustered.groupby("cluster").agg({
        "avg_basket_size": "mean",
        "discount_pct": "mean",
        "nps_score": "mean",
        "num_visits": "mean",
        "flash_sale_frequency": "mean",
        "customer_id": "count",
    }).round(1)
    cluster_stats.columns = ["Avg Basket", "Avg Discount %", "Avg NPS", "Avg Visits", "Flash Sale Freq", "Count"]

    personas = []
    for _, row in cluster_stats.iterrows():
        if row["Avg Basket"] >= cluster_stats["Avg Basket"].quantile(0.8):
            personas.append("Prestige Loyalist")
        elif row["Avg Discount %"] >= cluster_stats["Avg Discount %"].quantile(0.8):
            personas.append("Deal Hunter")
        elif row["Avg NPS"] >= cluster_stats["Avg NPS"].quantile(0.6):
            personas.append("Aspirational Buyer")
        elif row["Avg Visits"] <= cluster_stats["Avg Visits"].quantile(0.3):
            personas.append("Explorer")
        else:
            personas.append("Routine Professional")
    cluster_stats["Persona"] = personas
    clustered = clustered.merge(cluster_stats[["Persona"]], left_on="cluster", right_index=True, how="left")
    return clustered, cluster_stats.reset_index()


@st.cache_data
def run_association_rules(df):
    baskets = []
    for _, row in df.iterrows():
        basket = [row["category"], f"Brand_{row['brand_tier']}", f"Channel_{row['acquisition_channel']}"]
        if row["discount_pct"] > 30:
            basket.append("High_Discount")
        if row["flash_sale_response"] == 1:
            basket.append("Flash_Sale_Buyer")
        if row["repurchase_intent"] == 1:
            basket.append("High_Repurchase_Intent")
        baskets.append(basket)

    te = TransactionEncoder()
    basket_df = pd.DataFrame(te.fit_transform(baskets), columns=te.columns_)
    frequent_items = apriori(basket_df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=0.35)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    rules = rules.sort_values(["lift", "confidence"], ascending=False).head(12).copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)


@st.cache_data
def run_regression(df):
    X = df[["discount_pct", "months_active", "num_visits"]].values
    y = df["avg_basket_size"].values
    basket_model = LinearRegression().fit(X, y)

    discount_range = np.linspace(0, 70, 100)
    avg_months = df["months_active"].mean()
    avg_visits = df["num_visits"].mean()
    X_pred = np.column_stack([discount_range, np.full(100, avg_months), np.full(100, avg_visits)])
    basket_forecast = basket_model.predict(X_pred)

    nps_model = LinearRegression().fit(df[["discount_pct"]].values, df["nps_score"].values)
    nps_forecast = nps_model.predict(discount_range.reshape(-1, 1))
    return {
        "discount_range": discount_range,
        "basket_forecast": basket_forecast,
        "nps_forecast": nps_forecast,
        "basket_coef": round(basket_model.coef_[0], 1),
        "nps_coef": round(nps_model.coef_[0], 3),
        "r2_basket": round(basket_model.score(X, y), 3),
    }


with st.sidebar:
    st.title("✦ Tira Beauty")
    st.caption("Single-file Streamlit app with CSV-backed data")
    uploaded = st.file_uploader("Replace bundled CSV", type=["csv"])
    st.markdown("---")
    st.write("Use the bundled `tira_customers.csv` or upload a CSV with the same columns.")
    st.write("This version removes dependencies on `analytics.py` and `data_generator.py`.")


df = load_data(uploaded)
clf = run_classification(df)
df_clustered, cluster_stats = run_clustering(df)
rules = run_association_rules(df)
reg = run_regression(df)

st.title("Tira Beauty · Brand Intelligence Dashboard")
st.caption("Research question: Is discounting hurting premium brand perception?")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Customers", f"{len(df):,}")
with col2:
    st.metric("Avg discount", f"{df['discount_pct'].mean():.1f}%")
with col3:
    st.metric("Avg basket", f"₹{df['avg_basket_size'].mean():,.0f}")
with col4:
    st.metric("Premium loyal", f"{df['premium_loyal'].mean() * 100:.1f}%")

filters1, filters2, filters3 = st.columns(3)
with filters1:
    selected_tier = st.multiselect("Brand tier", sorted(df["brand_tier"].unique()), default=sorted(df["brand_tier"].unique()))
with filters2:
    selected_category = st.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
with filters3:
    max_discount = st.slider("Max discount %", 0, 80, 80)

filtered = df[(df["brand_tier"].isin(selected_tier)) & (df["category"].isin(selected_category)) & (df["discount_pct"] <= max_discount)].copy()
if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Customers", "ML Insights", "Rules & Forecast"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            filtered,
            x="discount_pct",
            y="nps_score",
            color="brand_tier",
            size="avg_basket_size",
            hover_data=["category", "acquisition_channel"],
            title="Discount exposure vs NPS",
            template=PLOTLY_TEMPLATE,
        )
        fig.add_vline(x=25, line_dash="dash", line_color=GOLD)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        revenue_by_group = filtered.groupby(["brand_tier", "category"], as_index=False)["estimated_revenue"].sum()
        fig = px.treemap(
            revenue_by_group,
            path=["brand_tier", "category"],
            values="estimated_revenue",
            color="estimated_revenue",
            color_continuous_scale="Sunsetdark",
            title="Estimated revenue by tier and category",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    monthly = filtered.groupby("discount_band", observed=False).agg(
        customers=("customer_id", "count"),
        avg_nps=("nps_score", "mean"),
        avg_basket=("avg_basket_size", "mean"),
    ).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["discount_band"], y=monthly["customers"], name="Customers", marker_color=BLUE), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly["discount_band"], y=monthly["avg_nps"], name="Avg NPS", mode="lines+markers", line=dict(color=GOLD, width=3)), secondary_y=True)
    fig.add_trace(go.Scatter(x=monthly["discount_band"], y=monthly["avg_basket"], name="Avg Basket", mode="lines+markers", line=dict(color=GREEN, width=3)), secondary_y=True)
    fig.update_layout(title="Performance across discount bands", template=PLOTLY_TEMPLATE)
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text="NPS / Basket", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.histogram(filtered, x="discount_pct", nbins=30, color="loyalty_label", barmode="overlay", title="Discount distribution by loyalty outcome", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        mix = filtered.groupby(["city_tier", "loyalty_label"], as_index=False)["customer_id"].count().rename(columns={"customer_id": "customers"})
        fig = px.bar(mix, x="city_tier", y="customers", color="loyalty_label", barmode="group", title="City tier mix", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        filtered[[
            "customer_id", "category", "brand_tier", "discount_pct", "nps_score",
            "avg_basket_size", "num_visits", "repurchase_intent", "loyalty_label"
        ]].sort_values(["discount_pct", "nps_score"], ascending=[False, True]),
        use_container_width=True,
        height=340,
    )

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        fi = clf["feature_importances"].reset_index()
        fi.columns = ["Feature", "Importance"]
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Classification feature importance", template=PLOTLY_TEMPLATE, color="Importance", color_continuous_scale="Tealgrn")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"""
            <div class='metric-card'>
                <b>Random Forest accuracy:</b> {clf['rf_accuracy']}%<br>
                <b>Logistic Regression accuracy:</b> {clf['lr_accuracy']}%<br>
                <b>Random Forest AUC:</b> {clf['rf_auc']}%
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        fig = px.scatter(
            df_clustered,
            x="discount_pct",
            y="avg_basket_size",
            color="Persona",
            hover_data=["category", "brand_tier", "nps_score"],
            title="Customer personas from clustering",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cluster_stats, use_container_width=True, height=290)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=reg["discount_range"], y=reg["basket_forecast"], mode="lines", name="Predicted basket", line=dict(color=BLUE, width=3)))
        fig.add_trace(go.Scatter(x=reg["discount_range"], y=reg["nps_forecast"], mode="lines", name="Predicted NPS", yaxis="y2", line=dict(color=GOLD, width=3)))
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Forecast effect of increasing discounts",
            yaxis=dict(title="Basket size"),
            yaxis2=dict(title="NPS", overlaying="y", side="right"),
            xaxis=dict(title="Discount %"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"""
            <div class='metric-card'>
                <b>Basket coefficient:</b> {reg['basket_coef']}<br>
                <b>NPS coefficient:</b> {reg['nps_coef']}<br>
                <b>Basket model R²:</b> {reg['r2_basket']}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        if rules.empty:
            st.info("No association rules met the confidence threshold.")
        else:
            top_rules = rules.copy()
            fig = px.bar(top_rules.head(10), x="lift", y="antecedents", color="confidence", orientation="h", title="Top association rules by lift", template=PLOTLY_TEMPLATE, color_continuous_scale="Sunset")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top_rules, use_container_width=True, height=300)

st.download_button(
    label="Download current dataset",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="tira_customers_filtered.csv",
    mime="text/csv",
)
