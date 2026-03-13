import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Tira Beauty · Brand Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_FILE = Path("tira_customers.csv")
REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "city_tier", "income_bracket", "acquisition_channel",
    "num_visits", "avg_basket_size", "discount_pct", "category", "brand_tier",
    "flash_sale_response", "flash_sale_frequency", "nps_score", "basket_trend",
    "willingness_full_price", "repurchase_intent", "months_active", "premium_loyal"
]

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#e5e7eb", "size": 12},
        "title": {"font": {"color": "#f9fafb", "size": 17}, "x": 0},
        "xaxis": {"gridcolor": "rgba(255,255,255,0.08)", "zeroline": False},
        "yaxis": {"gridcolor": "rgba(255,255,255,0.08)", "zeroline": False},
        "margin": {"l": 12, "r": 12, "t": 42, "b": 12},
        "colorway": ["#f59e0b", "#3b82f6", "#10b981", "#ec4899", "#8b5cf6", "#f97316"],
    }
}

st.markdown(
    """
    <style>
        .stApp {background: linear-gradient(180deg, #0b1220 0%, #111827 100%); color: #f9fafb;}
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1250px;}
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 10px 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error("CSV is missing required columns: " + ", ".join(missing))
        st.stop()


@st.cache_data(show_spinner=False)
def load_data(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    validate_columns(df)
    df = df.copy()
    df["estimated_revenue"] = df["avg_basket_size"] * df["num_visits"]
    df["loyalty_label"] = df["premium_loyal"].map({1: "Premium Loyal", 0: "Discount-Dependent"})
    df["discount_band"] = pd.cut(
        df["discount_pct"],
        bins=[-0.1, 10, 20, 30, 40, 100],
        labels=["0-10%", "10-20%", "20-30%", "30-40%", "40%+"],
    )
    return df


@st.cache_data(show_spinner=False)
def train_classifier(df: pd.DataFrame):
    features = ["age", "num_visits", "avg_basket_size", "discount_pct", "flash_sale_frequency", "nps_score", "months_active"]
    X = df[features]
    y = df["premium_loyal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return round(roc_auc_score(y_test, proba) * 100, 1), importance


@st.cache_data(show_spinner=False)
def run_clustering(df: pd.DataFrame):
    features = ["avg_basket_size", "discount_pct", "nps_score", "num_visits"]
    scaled = StandardScaler().fit_transform(df[features])
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled)
    out = df[["customer_id", "avg_basket_size", "discount_pct", "nps_score", "num_visits", "category", "brand_tier"]].copy()
    out["cluster"] = clusters
    summary = out.groupby("cluster", as_index=False).agg(
        customers=("customer_id", "count"),
        avg_basket_size=("avg_basket_size", "mean"),
        avg_discount=("discount_pct", "mean"),
        avg_nps=("nps_score", "mean"),
        avg_visits=("num_visits", "mean"),
    ).round(1)
    return out, summary


@st.cache_data(show_spinner=False)
def run_regression(df: pd.DataFrame):
    X = df[["discount_pct", "months_active", "num_visits"]]
    y = df["avg_basket_size"]
    model = LinearRegression().fit(X, y)
    discount_range = pd.Series(range(0, 71, 5), name="discount_pct")
    pred = pd.DataFrame({
        "discount_pct": discount_range,
        "months_active": df["months_active"].mean(),
        "num_visits": df["num_visits"].mean(),
    })
    pred["predicted_basket"] = model.predict(pred[["discount_pct", "months_active", "num_visits"]])
    return round(model.coef_[0], 1), round(model.score(X, y), 3), pred


if not DATA_FILE.exists():
    st.error("Missing tira_customers.csv in the same folder as app.py")
    st.stop()


df = load_data(str(DATA_FILE))

st.title("Tira Beauty · Brand Intelligence")
st.caption("Fast CSV-based dashboard with optional analytics on demand")

left, right = st.columns([1.2, 1])
with left:
    st.write("This version removes runtime data generation and defers heavier ML analysis until you ask for it.")
with right:
    st.download_button(
        "Download dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="tira_customers.csv",
        mime="text/csv",
    )

f1, f2, f3 = st.columns(3)
with f1:
    tier_filter = st.multiselect("Brand tier", sorted(df["brand_tier"].unique()), default=sorted(df["brand_tier"].unique()))
with f2:
    category_filter = st.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
with f3:
    max_discount = st.slider("Max discount", 0, int(df["discount_pct"].max()), int(df["discount_pct"].max()))

filtered = df[
    df["brand_tier"].isin(tier_filter)
    & df["category"].isin(category_filter)
    & (df["discount_pct"] <= max_discount)
].copy()

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(filtered):,}")
m2.metric("Avg discount", f"{filtered['discount_pct'].mean():.1f}%")
m3.metric("Avg basket", f"₹{filtered['avg_basket_size'].mean():,.0f}")
m4.metric("Premium loyal", f"{filtered['premium_loyal'].mean() * 100:.1f}%")

tab1, tab2, tab3 = st.tabs(["Overview", "Segments", "Optional ML"])

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
            title="Discount vs NPS",
            template=PLOTLY_TEMPLATE,
        )
        fig.add_vline(x=25, line_dash="dash", line_color="#f59e0b")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rev = filtered.groupby(["brand_tier", "category"], as_index=False)["estimated_revenue"].sum()
        fig = px.bar(
            rev.sort_values("estimated_revenue", ascending=False),
            x="category",
            y="estimated_revenue",
            color="brand_tier",
            title="Estimated revenue by category",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

    band = filtered.groupby("discount_band", observed=False).agg(
        customers=("customer_id", "count"),
        avg_nps=("nps_score", "mean"),
        avg_basket=("avg_basket_size", "mean"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=band["discount_band"], y=band["customers"], name="Customers"))
    fig.add_trace(go.Scatter(x=band["discount_band"], y=band["avg_nps"], mode="lines+markers", name="Avg NPS", yaxis="y2"))
    fig.add_trace(go.Scatter(x=band["discount_band"], y=band["avg_basket"], mode="lines+markers", name="Avg basket", yaxis="y2"))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Discount bands",
        yaxis=dict(title="Customers"),
        yaxis2=dict(title="NPS / Basket", overlaying="y", side="right"),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        city = filtered.groupby(["city_tier", "loyalty_label"], as_index=False)["customer_id"].count().rename(columns={"customer_id": "customers"})
        fig = px.bar(city, x="city_tier", y="customers", color="loyalty_label", barmode="group", title="City tier mix", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        mix = filtered.groupby(["category", "brand_tier"], as_index=False)["customer_id"].count().rename(columns={"customer_id": "customers"})
        fig = px.bar(mix, x="category", y="customers", color="brand_tier", title="Category mix", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        filtered[[
            "customer_id", "category", "brand_tier", "discount_pct", "nps_score",
            "avg_basket_size", "num_visits", "repurchase_intent", "loyalty_label"
        ]].sort_values(["discount_pct", "nps_score"], ascending=[False, True]),
        use_container_width=True,
        height=320,
    )

with tab3:
    st.write("These analyses are now optional and only run when you click a button.")

    a1, a2 = st.columns(2)
    with a1:
        if st.button("Run classifier", use_container_width=True):
            auc, importance = train_classifier(filtered)
            st.metric("Random Forest AUC", f"{auc}%")
            imp = importance.reset_index()
            imp.columns = ["feature", "importance"]
            fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Feature importance", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

    with a2:
        if st.button("Run clustering", use_container_width=True):
            clustered, summary = run_clustering(filtered)
            fig = px.scatter(
                clustered,
                x="discount_pct",
                y="avg_basket_size",
                color=clustered["cluster"].astype(str),
                hover_data=["brand_tier", "category", "nps_score"],
                title="Customer clusters",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(summary, use_container_width=True, height=220)

    if st.button("Run regression forecast", use_container_width=True):
        coef, r2, pred = run_regression(filtered)
        c1, c2 = st.columns(2)
        c1.metric("Discount coefficient", str(coef))
        c2.metric("R²", str(r2))
        fig = px.line(pred, x="discount_pct", y="predicted_basket", title="Predicted basket as discount rises", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
