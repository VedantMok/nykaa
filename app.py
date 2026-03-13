import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Tira Beauty · Analytics Dashboard", page_icon="✦", layout="wide")

DATA_FILE = Path("tira_customers.csv")
REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "city_tier", "income_bracket", "acquisition_channel",
    "num_visits", "avg_basket_size", "discount_pct", "category", "brand_tier",
    "flash_sale_response", "flash_sale_frequency", "nps_score", "basket_trend",
    "willingness_full_price", "repurchase_intent", "months_active", "premium_loyal"
]
COLORS = {
    "gold": "#f59e0b",
    "blue": "#3b82f6",
    "green": "#10b981",
    "pink": "#ec4899",
    "purple": "#8b5cf6",
    "orange": "#f97316",
    "bg": "#0f172a",
    "text": "#e2e8f0",
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=16, r=16, t=48, b=16),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #08111f 0%, #0f172a 100%);
            color: #e2e8f0;
        }
        .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1350px;}
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 18px;
            padding: 10px 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error("Missing required columns in tira_customers.csv: " + ", ".join(missing))
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
    net_factor = (1 - (df["discount_pct"] / 100)).clip(lower=0.2)
    df["full_price_basket"] = df["avg_basket_size"] / net_factor
    df["full_price_revenue"] = df["full_price_basket"] * df["num_visits"]
    df["discount_leakage"] = (df["full_price_revenue"] - df["estimated_revenue"]).clip(lower=0)
    df["loyalty_label"] = df["premium_loyal"].map({1: "Premium Loyal", 0: "Discount-Dependent"})
    df["discount_band"] = pd.cut(
        df["discount_pct"],
        bins=[-0.1, 10, 20, 30, 40, 100],
        labels=["0-10%", "10-20%", "20-30%", "30-40%", "40%+"],
    )
    df["basket_per_month"] = df["estimated_revenue"] / df["months_active"].clip(lower=1)
    return df


@st.cache_data(show_spinner=False)
def run_clustering(df: pd.DataFrame):
    features = ["avg_basket_size", "discount_pct", "nps_score", "num_visits", "flash_sale_frequency"]
    model_data = df[features].copy()
    scaled = StandardScaler().fit_transform(model_data)
    model = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)

    out = df.copy()
    out["cluster"] = labels
    stats = out.groupby("cluster").agg(
        customers=("customer_id", "count"),
        avg_basket=("avg_basket_size", "mean"),
        avg_discount=("discount_pct", "mean"),
        avg_nps=("nps_score", "mean"),
        avg_visits=("num_visits", "mean"),
        flash_freq=("flash_sale_frequency", "mean"),
        premium_loyal_rate=("premium_loyal", "mean"),
        revenue=("estimated_revenue", "mean"),
    ).round(2)

    personas = []
    recommendations = []
    for _, row in stats.iterrows():
        if row["avg_basket"] >= stats["avg_basket"].quantile(0.8):
            personas.append("Prestige Loyalist")
            recommendations.append("Protect with low-discount exclusives and early access launches")
        elif row["avg_discount"] >= stats["avg_discount"].quantile(0.8):
            personas.append("Deal Hunter")
            recommendations.append("Use targeted promotions and cap blanket discounting")
        elif row["avg_nps"] >= stats["avg_nps"].quantile(0.7):
            personas.append("Aspirational Buyer")
            recommendations.append("Upsell premium bundles and loyalty rewards")
        elif row["avg_visits"] <= stats["avg_visits"].quantile(0.3):
            personas.append("Explorer")
            recommendations.append("Use onboarding journeys and trial-size discovery packs")
        else:
            personas.append("Routine Professional")
            recommendations.append("Stabilize with subscriptions, replenishment reminders, and bundles")
    stats["persona"] = personas
    stats["target_action"] = recommendations
    stats["premium_loyal_rate"] = (stats["premium_loyal_rate"] * 100).round(1)
    out = out.merge(stats[["persona"]], left_on="cluster", right_index=True, how="left")
    return out, stats.reset_index()


@st.cache_data(show_spinner=False)
def run_association_rules(df: pd.DataFrame):
    baskets = []
    for _, row in df.iterrows():
        basket = [
            f"Category:{row['category']}",
            f"Brand:{row['brand_tier']}",
            f"Channel:{row['acquisition_channel']}",
            f"City:{row['city_tier']}",
        ]
        if row["discount_pct"] > 30:
            basket.append("Behavior:High Discount")
        if row["flash_sale_response"] == 1:
            basket.append("Behavior:Flash Sale")
        if row["repurchase_intent"] == 1:
            basket.append("Behavior:High Repurchase")
        if row["premium_loyal"] == 1:
            basket.append("Outcome:Premium Loyal")
        baskets.append(basket)

    te = TransactionEncoder()
    encoded = te.fit(baskets).transform(baskets)
    basket_df = pd.DataFrame(encoded, columns=te.columns_)
    frequent = apriori(basket_df, min_support=0.05, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    rules = association_rules(frequent, metric="confidence", min_threshold=0.35)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    rules = rules.sort_values(["lift", "confidence"], ascending=False).head(15).copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)


@st.cache_data(show_spinner=False)
def run_regression(df: pd.DataFrame):
    x_basket = df[["discount_pct", "months_active", "num_visits"]]
    y_basket = df["avg_basket_size"]
    basket_model = LinearRegression().fit(x_basket, y_basket)

    x_nps = df[["discount_pct", "months_active", "num_visits"]]
    y_nps = df["nps_score"]
    nps_model = LinearRegression().fit(x_nps, y_nps)

    discount_range = np.arange(0, 71, 5)
    avg_months = df["months_active"].mean()
    avg_visits = df["num_visits"].mean()
    pred = pd.DataFrame({
        "discount_pct": discount_range,
        "months_active": avg_months,
        "num_visits": avg_visits,
    })
    pred["predicted_basket"] = basket_model.predict(pred[["discount_pct", "months_active", "num_visits"]])
    pred["predicted_nps"] = nps_model.predict(pred[["discount_pct", "months_active", "num_visits"]])

    scenarios = pd.DataFrame({
        "discount_pct": [10, 20, 30, 40, 50],
        "months_active": avg_months,
        "num_visits": avg_visits,
    })
    scenarios["forecast_basket"] = basket_model.predict(scenarios[["discount_pct", "months_active", "num_visits"]]).round(0)
    scenarios["forecast_nps"] = nps_model.predict(scenarios[["discount_pct", "months_active", "num_visits"]]).round(2)
    return {
        "forecast": pred,
        "scenarios": scenarios,
        "basket_discount_coef": round(basket_model.coef_[0], 2),
        "nps_discount_coef": round(nps_model.coef_[0], 3),
        "basket_r2": round(basket_model.score(x_basket, y_basket), 3),
        "nps_r2": round(nps_model.score(x_nps, y_nps), 3),
    }


def build_sankey(df: pd.DataFrame):
    stage1 = df.groupby(["acquisition_channel", "city_tier"]).size().reset_index(name="value")
    stage2 = df.groupby(["city_tier", "loyalty_label"]).size().reset_index(name="value")
    labels = list(dict.fromkeys(
        stage1["acquisition_channel"].tolist() +
        stage1["city_tier"].tolist() +
        stage2["loyalty_label"].tolist()
    ))
    idx = {label: i for i, label in enumerate(labels)}
    source = [idx[s] for s in stage1["acquisition_channel"]] + [idx[s] for s in stage2["city_tier"]]
    target = [idx[t] for t in stage1["city_tier"]] + [idx[t] for t in stage2["loyalty_label"]]
    value = stage1["value"].tolist() + stage2["value"].tolist()
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18, thickness=18,
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            label=labels,
            color=[COLORS["blue"], COLORS["green"], COLORS["gold"], COLORS["pink"], COLORS["purple"]] * 3,
        ),
        link=dict(source=source, target=target, value=value)
    ))
    fig.update_layout(title="Customer flow: acquisition → city tier → loyalty", **PLOTLY_LAYOUT)
    return fig


def build_waterfall(df: pd.DataFrame):
    full_price = df["full_price_revenue"].sum()
    leakage = df["discount_leakage"].sum()
    actual = df["estimated_revenue"].sum()
    fig = go.Figure(go.Waterfall(
        name="Revenue",
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Full-price potential", "Discount leakage", "Realized revenue"],
        y=[full_price, -leakage, actual],
        connector={"line": {"color": "rgba(255,255,255,0.25)"}},
        increasing={"marker": {"color": COLORS["green"]}},
        decreasing={"marker": {"color": COLORS["pink"]}},
        totals={"marker": {"color": COLORS["blue"]}},
    ))
    fig.update_layout(title="Waterfall: revenue impact of discount leakage", **PLOTLY_LAYOUT)
    return fig


def build_radar(cluster_stats: pd.DataFrame):
    scaled = cluster_stats.copy()
    metric_cols = ["avg_basket", "avg_discount", "avg_nps", "avg_visits", "flash_freq"]
    for col in metric_cols:
        cmin, cmax = scaled[col].min(), scaled[col].max()
        scaled[col] = 1 if cmax == cmin else (scaled[col] - cmin) / (cmax - cmin)
    fig = go.Figure()
    for i, row in scaled.iterrows():
        values = [row[c] for c in metric_cols]
        values += values[:1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=["Basket", "Discount", "NPS", "Visits", "Flash Sale"] * 1 + ["Basket"],
            fill="toself",
            name=cluster_stats.loc[i, "persona"],
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.15)")),
        title="Persona radar comparison",
        **PLOTLY_LAYOUT,
    )
    return fig


def build_pydeck(df: pd.DataFrame):
    anchors = {
        "Tier 1": {"lat": 19.0760, "lon": 72.8777},
        "Tier 2": {"lat": 17.3850, "lon": 78.4867},
        "Tier 3": {"lat": 26.9124, "lon": 75.7873},
    }
    summary = df.groupby("city_tier").agg(
        customers=("customer_id", "count"),
        avg_discount=("discount_pct", "mean"),
        avg_nps=("nps_score", "mean"),
    ).reset_index()
    summary["lat"] = summary["city_tier"].map(lambda x: anchors.get(x, {}).get("lat", 21.0))
    summary["lon"] = summary["city_tier"].map(lambda x: anchors.get(x, {}).get("lon", 78.0))
    summary["elevation"] = summary["customers"] * 80
    layer = pdk.Layer(
        "ColumnLayer",
        data=summary,
        get_position="[lon, lat]",
        get_elevation="elevation",
        elevation_scale=1,
        radius=34000,
        pickable=True,
        auto_highlight=True,
        extruded=True,
        get_fill_color=[59, 130, 246, 180],
    )
    view_state = pdk.ViewState(latitude=21.5, longitude=78.5, zoom=4, pitch=45)
    tooltip = {"html": "<b>{city_tier}</b><br/>Customers: {customers}<br/>Avg discount: {avg_discount:.1f}%<br/>Avg NPS: {avg_nps:.1f}"}
    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="light", tooltip=tooltip)


def descriptive_insights(df: pd.DataFrame):
    top_category = df.groupby("category")["estimated_revenue"].sum().sort_values(ascending=False).index[0]
    top_channel = df["acquisition_channel"].value_counts().index[0]
    return [
        f"{top_category} contributes the highest estimated revenue in the current filtered view.",
        f"{top_channel} is the largest acquisition channel in the filtered customer base.",
        f"Average discount is {df['discount_pct'].mean():.1f}% while premium-loyal share is {df['premium_loyal'].mean() * 100:.1f}%.",
    ]


def prescriptive_recommendations(cluster_stats: pd.DataFrame, rules: pd.DataFrame, reg: dict):
    recs = []
    strongest_persona = cluster_stats.sort_values("premium_loyal_rate", ascending=False).iloc[0]
    recs.append(f"Prioritize {strongest_persona['persona']} because it has the highest premium-loyal rate and stronger basket economics.")
    if not rules.empty:
        top_rule = rules.iloc[0]
        recs.append(f"Use {top_rule['antecedents']} as a trigger set because it most strongly links to {top_rule['consequents']}.")
    if reg["basket_discount_coef"] < 0:
        recs.append("Cap broad discounting and move to targeted offers, because the basket forecast declines as discount intensity rises.")
    recs.append("Build separate campaigns for Deal Hunters, Explorers, and Prestige Loyalists instead of one common promotion plan.")
    return recs


df = load_data()

with st.sidebar:
    st.title("✦ Tira Beauty")
    st.caption("Assignment dashboard with descriptive, diagnostic, predictive, and prescriptive views")
    tiers = sorted(df["brand_tier"].unique())
    categories = sorted(df["category"].unique())
    cities = sorted(df["city_tier"].unique())
    channels = sorted(df["acquisition_channel"].unique())

    selected_tiers = st.multiselect("Brand tier", tiers, default=tiers)
    selected_categories = st.multiselect("Category", categories, default=categories)
    selected_cities = st.multiselect("City tier", cities, default=cities)
    selected_channels = st.multiselect("Channel", channels, default=channels)
    discount_range = st.slider("Discount range %", 0, int(df["discount_pct"].max()), (0, int(df["discount_pct"].max())))

filtered = df[
    df["brand_tier"].isin(selected_tiers)
    & df["category"].isin(selected_categories)
    & df["city_tier"].isin(selected_cities)
    & df["acquisition_channel"].isin(selected_channels)
    & df["discount_pct"].between(discount_range[0], discount_range[1])
].copy()

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

clustered_df, cluster_stats = run_clustering(filtered)
rules = run_association_rules(filtered)
reg = run_regression(filtered)

st.title("Tira Beauty · Brand Intelligence Dashboard")
st.caption("Research question: Is discounting hurting premium brand perception?")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(filtered):,}")
m2.metric("Avg basket", f"₹{filtered['avg_basket_size'].mean():,.0f}")
m3.metric("Avg discount", f"{filtered['discount_pct'].mean():.1f}%")
m4.metric("Premium loyal", f"{filtered['premium_loyal'].mean() * 100:.1f}%")

descriptive_tab, diagnostic_tab, predictive_tab, prescriptive_tab = st.tabs(
    ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
)

with descriptive_tab:
    st.subheader("What is happening?")
    desc1, desc2 = st.columns(2)
    with desc1:
        revenue_tree = filtered.groupby(["brand_tier", "category"], as_index=False)["estimated_revenue"].sum()
        fig = px.treemap(
            revenue_tree,
            path=["brand_tier", "category"],
            values="estimated_revenue",
            color="estimated_revenue",
            color_continuous_scale="Sunsetdark",
            title="Treemap: revenue by brand tier and category",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with desc2:
        st.plotly_chart(build_sankey(filtered), use_container_width=True)

    drill1, drill2 = st.columns(2)
    with drill1:
        drill = filtered.groupby(["brand_tier", "category", "acquisition_channel"], as_index=False)["estimated_revenue"].sum()
        fig = px.sunburst(
            drill,
            path=["brand_tier", "category", "acquisition_channel"],
            values="estimated_revenue",
            title="Drill-down: tier → category → channel",
            color="estimated_revenue",
            color_continuous_scale="Blues",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with drill2:
        st.pydeck_chart(build_pydeck(filtered), use_container_width=True)

    for insight in descriptive_insights(filtered):
        st.info(insight)

with diagnostic_tab:
    st.subheader("Why is it happening?")
    d1, d2 = st.columns(2)
    with d1:
        fig = px.scatter(
            filtered,
            x="discount_pct",
            y="nps_score",
            color="brand_tier",
            size="avg_basket_size",
            hover_data=["category", "acquisition_channel", "city_tier"],
            title="Discount exposure vs NPS",
            color_discrete_sequence=[COLORS["gold"], COLORS["blue"], COLORS["green"]],
        )
        fig.add_vline(x=25, line_dash="dash", line_color=COLORS["gold"])
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        fig = px.scatter(
            filtered,
            x="discount_pct",
            y="avg_basket_size",
            color="loyalty_label",
            hover_data=["category", "brand_tier", "city_tier"],
            title="Discount exposure vs basket size",
            color_discrete_sequence=[COLORS["green"], COLORS["pink"]],
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    d3, d4 = st.columns(2)
    with d3:
        st.plotly_chart(build_waterfall(filtered), use_container_width=True)
    with d4:
        heat = pd.pivot_table(
            filtered,
            index="category",
            columns="discount_band",
            values="nps_score",
            aggfunc="mean",
        ).fillna(0)
        fig = px.imshow(
            heat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Tealgrn",
            title="Heatmap: average NPS by category and discount band",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

with predictive_tab:
    st.subheader("What is likely to happen next?")
    p1, p2 = st.columns([2, 1])
    with p1:
        pred = reg["forecast"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=pred["discount_pct"], y=pred["predicted_basket"], name="Predicted basket", line=dict(color=COLORS["blue"], width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=pred["discount_pct"], y=pred["predicted_nps"], name="Predicted NPS", line=dict(color=COLORS["gold"], width=3)), secondary_y=True)
        fig.update_layout(title="Regression forecast as discounting rises", **PLOTLY_LAYOUT)
        fig.update_yaxes(title_text="Basket size", secondary_y=False)
        fig.update_yaxes(title_text="NPS", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        st.metric("Basket discount coefficient", reg["basket_discount_coef"])
        st.metric("NPS discount coefficient", reg["nps_discount_coef"])
        st.metric("Basket R²", reg["basket_r2"])
        st.metric("NPS R²", reg["nps_r2"])

    st.dataframe(reg["scenarios"], use_container_width=True, hide_index=True)

with prescriptive_tab:
    st.subheader("What should the business do?")
    pr1, pr2 = st.columns(2)
    with pr1:
        fig = px.scatter(
            clustered_df,
            x="discount_pct",
            y="avg_basket_size",
            color="persona",
            hover_data=["category", "brand_tier", "nps_score", "num_visits"],
            title="Clustering: customer personas and target segments",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with pr2:
        st.plotly_chart(build_radar(cluster_stats), use_container_width=True)

    st.dataframe(cluster_stats, use_container_width=True, hide_index=True)

    st.markdown("### Association rule mining")
    if rules.empty:
        st.info("No strong rules were found for the current filter selection.")
    else:
        rule_fig = px.bar(
            rules.head(10),
            x="lift",
            y="antecedents",
            color="confidence",
            orientation="h",
            title="Top association rules by lift",
            color_continuous_scale="Sunset",
        )
        rule_fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(rule_fig, use_container_width=True)
        st.dataframe(rules, use_container_width=True, hide_index=True)

    st.markdown("### Recommended actions")
    for rec in prescriptive_recommendations(cluster_stats, rules, reg):
        st.success(rec)

st.download_button(
    "Download filtered dataset",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="tira_customers_filtered.csv",
    mime="text/csv",
)
