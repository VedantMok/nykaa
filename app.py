import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Tira Dashboard", page_icon="📊", layout="wide")

DATA_FILE = Path("tira_customers.csv")
REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "city_tier", "income_bracket", "acquisition_channel",
    "num_visits", "avg_basket_size", "discount_pct", "category", "brand_tier",
    "flash_sale_response", "flash_sale_frequency", "nps_score", "basket_trend",
    "willingness_full_price", "repurchase_intent", "months_active", "premium_loyal"
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error("Missing columns in tira_customers.csv: " + ", ".join(missing))
        st.stop()


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error("tira_customers.csv was not found in the same folder as app.py")
        st.stop()
    df = pd.read_csv(DATA_FILE)
    validate_columns(df)
    df = df.copy()
    df["estimated_revenue"] = df["avg_basket_size"] * df["num_visits"]
    df["loyalty_label"] = df["premium_loyal"].map({1: "Premium Loyal", 0: "Discount-Dependent"})
    return df


df = load_data()

st.title("Tira Beauty Dashboard")
st.caption("Minimal version for reliable Streamlit deployment")

with st.sidebar:
    st.header("Filters")
    categories = sorted(df["category"].dropna().unique().tolist())
    tiers = sorted(df["brand_tier"].dropna().unique().tolist())
    cities = sorted(df["city_tier"].dropna().unique().tolist())

    selected_categories = st.multiselect("Category", categories, default=categories)
    selected_tiers = st.multiselect("Brand tier", tiers, default=tiers)
    selected_cities = st.multiselect("City tier", cities, default=cities)
    max_discount = st.slider(
        "Max discount %",
        min_value=0,
        max_value=int(df["discount_pct"].max()),
        value=int(df["discount_pct"].max()),
    )

filtered = df[
    df["category"].isin(selected_categories)
    & df["brand_tier"].isin(selected_tiers)
    & df["city_tier"].isin(selected_cities)
    & (df["discount_pct"] <= max_discount)
].copy()

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{len(filtered):,}")
col2.metric("Avg basket", f"₹{filtered['avg_basket_size'].mean():,.0f}")
col3.metric("Avg discount", f"{filtered['discount_pct'].mean():.1f}%")
col4.metric("Premium loyal", f"{filtered['premium_loyal'].mean() * 100:.1f}%")

st.subheader("Discount and perception")
chart_df = (
    filtered.groupby("category", as_index=False)
    .agg(avg_discount=("discount_pct", "mean"), avg_nps=("nps_score", "mean"), avg_basket=("avg_basket_size", "mean"))
    .sort_values("avg_discount", ascending=False)
)
left, right = st.columns(2)
with left:
    st.write("Average discount by category")
    st.bar_chart(chart_df.set_index("category")[["avg_discount"]])
with right:
    st.write("Average NPS by category")
    st.bar_chart(chart_df.set_index("category")[["avg_nps"]])

st.subheader("Brand mix")
brand_mix = (
    filtered.groupby(["brand_tier", "loyalty_label"], as_index=False)
    .size()
    .rename(columns={"size": "customers"})
)
st.dataframe(brand_mix, use_container_width=True, hide_index=True)

st.subheader("Revenue snapshot")
revenue_df = (
    filtered.groupby("brand_tier", as_index=False)
    .agg(estimated_revenue=("estimated_revenue", "sum"))
    .sort_values("estimated_revenue", ascending=False)
)
st.bar_chart(revenue_df.set_index("brand_tier")[["estimated_revenue"]])

st.subheader("Customer view")
st.dataframe(
    filtered[[
        "customer_id", "category", "brand_tier", "city_tier", "discount_pct",
        "nps_score", "avg_basket_size", "num_visits", "loyalty_label"
    ]].sort_values(["discount_pct", "nps_score"], ascending=[False, True]),
    use_container_width=True,
    hide_index=True,
    height=360,
)

st.download_button(
    "Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="tira_customers_filtered.csv",
    mime="text/csv",
)
