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

st.set_page_config(page_title="Tira Beauty · Brand Intelligence", page_icon="✦", layout="wide")

DATA_FILE = Path("tira_customers.csv")
REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "city_tier", "income_bracket", "acquisition_channel",
    "num_visits", "avg_basket_size", "discount_pct", "category", "brand_tier",
    "flash_sale_response", "flash_sale_frequency", "nps_score", "basket_trend",
    "willingness_full_price", "repurchase_intent", "months_active", "premium_loyal"
]
THEME = {
    "bg": "#070b12",
    "panel": "#0f1623",
    "panel2": "#121b2b",
    "border": "rgba(255,255,255,0.08)",
    "text": "#f5f7fb",
    "muted": "#94a3b8",
    "gold": "#d4a843",
    "blue": "#4f8cff",
    "green": "#2bc48a",
    "rose": "#ef6b8a",
    "purple": "#9b7bff",
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=THEME["text"], family="Inter, sans-serif", size=12),
    margin=dict(l=18, r=18, t=52, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(212,168,67,0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(79,140,255,0.10), transparent 24%),
                linear-gradient(180deg, #060910 0%, #0b1220 55%, #0f172a 100%);
            color: {THEME['text']};
        }}
        .block-container {{
            max-width: 1360px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(15,22,35,0.98) 0%, rgba(11,17,28,0.98) 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        h1, h2, h3, h4 {{ color: {THEME['text']}; letter-spacing: -0.02em; }}
        .hero {{
            background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 24px 26px 20px 26px;
            margin-bottom: 1rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.25);
        }}
        .eyebrow {{
            color: {THEME['gold']};
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.18em;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        .hero-title {{
            font-size: 2.25rem;
            font-weight: 700;
            line-height: 1.05;
            margin: 0 0 8px 0;
        }}
        .hero-sub {{
            color: {THEME['muted']};
            font-size: 1rem;
            max-width: 820px;
            margin-bottom: 0;
        }}
        .section-card {{
            background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.025) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 18px 18px 12px 18px;
            margin-bottom: 16px;
        }}
        .insight-card {{
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            padding: 14px 16px;
            min-height: 96px;
        }}
        .insight-title {{
            color: {THEME['gold']};
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 6px;
        }}
        .insight-body {{
            color: {THEME['text']};
            font-size: 0.95rem;
            line-height: 1.45;
        }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.025));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 12px 16px;
        }}
        div[data-testid="stMetric"] label {{ color: {THEME['muted']} !important; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 999px;
            height: 44px;
            padding: 0 16px;
            color: {THEME['muted']};
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(180deg, rgba(212,168,67,0.18), rgba(212,168,67,0.08));
            color: white !important;
            border-color: rgba(212,168,67,0.35);
        }}
        .small-note {{ color: {THEME['muted']}; font-size: 0.88rem; margin-top: -4px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def chart_layout(fig, title=None):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False, linecolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False, linecolor="rgba(255,255,255,0.12)")
    return fig


def card_start(title, subtitle=None):
    subtitle_html = f'<div class="small-note">{subtitle}</div>' if subtitle else ''
    st.markdown(f'<div class="section-card"><h3 style="margin:0 0 6px 0;">{title}</h3>{subtitle_html}', unsafe_allow_html=True)


def card_end():
    st.markdown('</div>', unsafe_allow_html=True)


def insight_box(title, body):
    st.markdown(
        f'<div class="insight-card"><div class="insight-title">{title}</div><div class="insight-body">{body}</div></div>',
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
    df["discount_band"] = pd.cut(df["discount_pct"], bins=[-0.1, 10, 20, 30, 40, 100], labels=["0-10%", "10-20%", "20-30%", "30-40%", "40%+"])
    df["basket_per_month"] = df["estimated_revenue"] / df["months_active"].clip(lower=1)
    return df


@st.cache_data(show_spinner=False)
def run_clustering(df: pd.DataFrame):
    features = ["avg_basket_size", "discount_pct", "nps_score", "num_visits", "flash_sale_frequency"]
    scaled = StandardScaler().fit_transform(df[features])
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
    personas, actions = [], []
    for _, row in stats.iterrows():
        if row["avg_basket"] >= stats["avg_basket"].quantile(0.8):
            personas.append("Prestige Loyalist")
            actions.append("Protect with low-discount exclusives and early-access launches")
        elif row["avg_discount"] >= stats["avg_discount"].quantile(0.8):
            personas.append("Deal Hunter")
            actions.append("Use tighter targeted promotions instead of blanket discounts")
        elif row["avg_nps"] >= stats["avg_nps"].quantile(0.7):
            personas.append("Aspirational Buyer")
            actions.append("Upsell premium bundles and loyalty rewards")
        elif row["avg_visits"] <= stats["avg_visits"].quantile(0.3):
            personas.append("Explorer")
            actions.append("Guide discovery through onboarding and trial packs")
        else:
            personas.append("Routine Professional")
            actions.append("Improve retention with replenishment reminders and subscriptions")
    stats["persona"] = personas
    stats["target_action"] = actions
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
    rules = rules.sort_values(["lift", "confidence"], ascending=False).head(12).copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)


@st.cache_data(show_spinner=False)
def run_regression(df: pd.DataFrame):
    x = df[["discount_pct", "months_active", "num_visits"]]
    basket_model = LinearRegression().fit(x, df["avg_basket_size"])
    nps_model = LinearRegression().fit(x, df["nps_score"])
    discount_range = np.arange(0, 71, 5)
    avg_months = df["months_active"].mean()
    avg_visits = df["num_visits"].mean()
    pred = pd.DataFrame({"discount_pct": discount_range, "months_active": avg_months, "num_visits": avg_visits})
    pred["predicted_basket"] = basket_model.predict(pred[["discount_pct", "months_active", "num_visits"]])
    pred["predicted_nps"] = nps_model.predict(pred[["discount_pct", "months_active", "num_visits"]])
    scenarios = pd.DataFrame({"discount_pct": [10, 20, 30, 40, 50], "months_active": avg_months, "num_visits": avg_visits})
    scenarios["forecast_basket"] = basket_model.predict(scenarios[["discount_pct", "months_active", "num_visits"]]).round(0)
    scenarios["forecast_nps"] = nps_model.predict(scenarios[["discount_pct", "months_active", "num_visits"]]).round(2)
    return {
        "forecast": pred,
        "scenarios": scenarios,
        "basket_discount_coef": round(basket_model.coef_[0], 2),
        "nps_discount_coef": round(nps_model.coef_[0], 3),
        "basket_r2": round(basket_model.score(x, df["avg_basket_size"]), 3),
        "nps_r2": round(nps_model.score(x, df["nps_score"]), 3),
    }


def build_sankey(df: pd.DataFrame):
    stage1 = df.groupby(["acquisition_channel", "city_tier"]).size().reset_index(name="value")
    stage2 = df.groupby(["city_tier", "loyalty_label"]).size().reset_index(name="value")
    labels = list(dict.fromkeys(stage1["acquisition_channel"].tolist() + stage1["city_tier"].tolist() + stage2["loyalty_label"].tolist()))
    idx = {label: i for i, label in enumerate(labels)}
    fig = go.Figure(go.Sankey(
        node=dict(pad=18, thickness=18, label=labels, line=dict(color="rgba(255,255,255,0.08)", width=1), color=[THEME["gold"], THEME["blue"], THEME["green"], THEME["rose"], THEME["purple"]] * 3),
        link=dict(source=[idx[s] for s in stage1["acquisition_channel"]] + [idx[s] for s in stage2["city_tier"]], target=[idx[t] for t in stage1["city_tier"]] + [idx[t] for t in stage2["loyalty_label"]], value=stage1["value"].tolist() + stage2["value"].tolist(), color="rgba(255,255,255,0.12)"),
    ))
    return chart_layout(fig, "Customer flow")


def build_waterfall(df: pd.DataFrame):
    fig = go.Figure(go.Waterfall(
        measure=["absolute", "relative", "total"],
        x=["Full-price potential", "Discount leakage", "Realized revenue"],
        y=[df["full_price_revenue"].sum(), -df["discount_leakage"].sum(), df["estimated_revenue"].sum()],
        connector={"line": {"color": "rgba(255,255,255,0.2)"}},
        increasing={"marker": {"color": THEME["green"]}},
        decreasing={"marker": {"color": THEME["rose"]}},
        totals={"marker": {"color": THEME["blue"]}},
    ))
    return chart_layout(fig, "Revenue impact of discount leakage")


def build_radar(cluster_stats: pd.DataFrame):
    scaled = cluster_stats.copy()
    cols = ["avg_basket", "avg_discount", "avg_nps", "avg_visits", "flash_freq"]
    for col in cols:
        cmin, cmax = scaled[col].min(), scaled[col].max()
        scaled[col] = 1 if cmax == cmin else (scaled[col] - cmin) / (cmax - cmin)
    fig = go.Figure()
    theta = ["Basket", "Discount", "NPS", "Visits", "Flash Sale", "Basket"]
    for i, row in scaled.iterrows():
        vals = [row[c] for c in cols]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(r=vals, theta=theta, fill="toself", name=cluster_stats.loc[i, "persona"]))
    fig.update_layout(**PLOTLY_LAYOUT, polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.12)")), title="Persona profile comparison")
    return fig


def build_pydeck(df: pd.DataFrame):
    anchors = {"Tier 1": {"lat": 19.0760, "lon": 72.8777}, "Tier 2": {"lat": 17.3850, "lon": 78.4867}, "Tier 3": {"lat": 26.9124, "lon": 75.7873}}
    summary = df.groupby("city_tier").agg(customers=("customer_id", "count"), avg_discount=("discount_pct", "mean"), avg_nps=("nps_score", "mean")).reset_index()
    summary["lat"] = summary["city_tier"].map(lambda x: anchors.get(x, {}).get("lat", 21.0))
    summary["lon"] = summary["city_tier"].map(lambda x: anchors.get(x, {}).get("lon", 78.0))
    summary["elevation"] = summary["customers"] * 80
    layer = pdk.Layer("ColumnLayer", data=summary, get_position="[lon, lat]", get_elevation="elevation", elevation_scale=1, radius=34000, pickable=True, auto_highlight=True, extruded=True, get_fill_color=[79, 140, 255, 180])
    return pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=21.5, longitude=78.5, zoom=4, pitch=42), map_style="light", tooltip={"html": "<b>{city_tier}</b><br/>Customers: {customers}<br/>Avg discount: {avg_discount:.1f}%<br/>Avg NPS: {avg_nps:.1f}"})


def descriptive_insights(df: pd.DataFrame):
    top_category = df.groupby("category")["estimated_revenue"].sum().sort_values(ascending=False).index[0]
    top_channel = df["acquisition_channel"].value_counts().index[0]
    top_tier = df.groupby("brand_tier")["estimated_revenue"].sum().sort_values(ascending=False).index[0]
    return [
        ("Lead segment", f"{top_tier} is the strongest revenue tier, with {top_category} leading the category mix."),
        ("Acquisition", f"{top_channel} is the largest acquisition channel in the filtered audience."),
        ("Price perception", f"Average discount is {df['discount_pct'].mean():.1f}% while premium-loyal share is {df['premium_loyal'].mean() * 100:.1f}%.")
    ]


def prescriptive_recommendations(cluster_stats: pd.DataFrame, rules: pd.DataFrame, reg: dict):
    recs = []
    strongest = cluster_stats.sort_values("premium_loyal_rate", ascending=False).iloc[0]
    recs.append(f"Prioritize {strongest['persona']} with premium-first journeys, because it has the highest loyalty quality.")
    if not rules.empty:
        top = rules.iloc[0]
        recs.append(f"Use {top['antecedents']} as a trigger set, since it strongly connects to {top['consequents']}.")
    if reg["basket_discount_coef"] < 0:
        recs.append("Cap broad discounting and move toward targeted offers, because higher discounts forecast weaker basket value.")
    recs.append("Separate CRM plays for Deal Hunters, Explorers, and Prestige Loyalists instead of one common campaign.")
    return recs


def filtered_kpi_label(df: pd.DataFrame):
    return f"{len(df):,} customers · {df['brand_tier'].nunique()} tiers · {df['category'].nunique()} categories"


df = load_data()
with st.sidebar:
    st.title("✦ Tira Beauty")
    st.caption("Luxury brand dashboard · assignment edition")
    st.markdown("---")
    st.markdown("**Filters**")
    tiers = sorted(df["brand_tier"].unique())
    categories = sorted(df["category"].unique())
    cities = sorted(df["city_tier"].unique())
    channels = sorted(df["acquisition_channel"].unique())
    selected_tiers = st.multiselect("Brand tier", tiers, default=tiers)
    selected_categories = st.multiselect("Category", categories, default=categories)
    selected_cities = st.multiselect("City tier", cities, default=cities)
    selected_channels = st.multiselect("Channel", channels, default=channels)
    discount_range = st.slider("Discount range %", 0, int(df["discount_pct"].max()), (0, int(df["discount_pct"].max())))
    st.markdown("---")
    st.caption("Use filters to create drill-down views across tier, category, city, and acquisition channel.")

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

hero_html = f"""
<div class=\"hero\">
  <div class=\"eyebrow\">Brand Intelligence Dashboard</div>
  <div class=\"hero-title\">Tira Beauty</div>
  <p class=\"hero-sub\">A cleaner executive dashboard to explain what is happening, why it is happening, what is likely next, and what the brand should do about it. {filtered_kpi_label(filtered)}</p>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(filtered):,}")
m2.metric("Avg basket", f"₹{filtered['avg_basket_size'].mean():,.0f}")
m3.metric("Avg discount", f"{filtered['discount_pct'].mean():.1f}%")
m4.metric("Premium loyal", f"{filtered['premium_loyal'].mean() * 100:.1f}%")

insights = descriptive_insights(filtered)
ci1, ci2, ci3 = st.columns(3)
with ci1:
    insight_box(*insights[0])
with ci2:
    insight_box(*insights[1])
with ci3:
    insight_box(*insights[2])

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
overview_tab, drivers_tab, forecast_tab, actions_tab = st.tabs(["Overview", "Drivers", "Forecast", "Actions"])

with overview_tab:
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        card_start("Revenue composition", "Treemap highlights where revenue concentrates across brand tiers and categories.")
        tree = filtered.groupby(["brand_tier", "category"], as_index=False)["estimated_revenue"].sum()
        fig = px.treemap(tree, path=["brand_tier", "category"], values="estimated_revenue", color="estimated_revenue", color_continuous_scale="Sunsetdark")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c2:
        card_start("Customer flow", "Sankey view simplifies channel-to-tier-to-loyalty movement.")
        st.plotly_chart(build_sankey(filtered), use_container_width=True)
        card_end()

    c3, c4 = st.columns([1.05, 0.95])
    with c3:
        card_start("Drill-down hierarchy", "Use this sunburst as a fast drill-down from tier to category to channel.")
        drill = filtered.groupby(["brand_tier", "category", "acquisition_channel"], as_index=False)["estimated_revenue"].sum()
        fig = px.sunburst(drill, path=["brand_tier", "category", "acquisition_channel"], values="estimated_revenue", color="estimated_revenue", color_continuous_scale="Blues")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c4:
        card_start("Market footprint", "City-tier map is illustrative and keeps the geo story visually lightweight.")
        st.pydeck_chart(build_pydeck(filtered), use_container_width=True)
        card_end()

with drivers_tab:
    c1, c2 = st.columns(2)
    with c1:
        card_start("Discount vs NPS", "Brand perception softens as discount intensity increases.")
        fig = px.scatter(filtered, x="discount_pct", y="nps_score", color="brand_tier", size="avg_basket_size", hover_data=["category", "acquisition_channel", "city_tier"], color_discrete_sequence=[THEME["gold"], THEME["blue"], THEME["green"]])
        fig.add_vline(x=25, line_dash="dash", line_color=THEME["gold"])
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c2:
        card_start("Discount leakage", "Waterfall shows how much potential revenue is given up through discounting.")
        st.plotly_chart(build_waterfall(filtered), use_container_width=True)
        card_end()

    c3, c4 = st.columns([1.05, 0.95])
    with c3:
        card_start("Basket sensitivity", "This plot shows how basket size shifts across loyalty outcomes and discount levels.")
        fig = px.scatter(filtered, x="discount_pct", y="avg_basket_size", color="loyalty_label", hover_data=["category", "brand_tier", "city_tier"], color_discrete_sequence=[THEME["green"], THEME["rose"]])
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c4:
        card_start("Heatmap", "Average NPS by category and discount band makes pressure points easier to spot.")
        heat = pd.pivot_table(filtered, index="category", columns="discount_band", values="nps_score", aggfunc="mean").fillna(0)
        fig = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="Tealgrn")
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        card_end()

with forecast_tab:
    c1, c2 = st.columns([1.35, 0.65])
    with c1:
        card_start("Regression forecast", "Forecast lines estimate how rising discounts affect basket value and NPS.")
        pred = reg["forecast"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=pred["discount_pct"], y=pred["predicted_basket"], name="Predicted basket", line=dict(color=THEME["blue"], width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=pred["discount_pct"], y=pred["predicted_nps"], name="Predicted NPS", line=dict(color=THEME["gold"], width=3)), secondary_y=True)
        chart_layout(fig, "Forecast as discounts rise")
        fig.update_yaxes(title_text="Basket size", secondary_y=False)
        fig.update_yaxes(title_text="NPS", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c2:
        card_start("Model readout", "Compact diagnostics keep the view executive-friendly.")
        st.metric("Basket discount coefficient", reg["basket_discount_coef"])
        st.metric("NPS discount coefficient", reg["nps_discount_coef"])
        st.metric("Basket R²", reg["basket_r2"])
        st.metric("NPS R²", reg["nps_r2"])
        card_end()

    card_start("Scenario table", "A simple scenario strip makes the forecast easier to present in class.")
    st.dataframe(reg["scenarios"], use_container_width=True, hide_index=True)
    card_end()

with actions_tab:
    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        card_start("Customer personas", "Clustering separates the base into targetable personas rather than one blended audience.")
        fig = px.scatter(clustered_df, x="discount_pct", y="avg_basket_size", color="persona", hover_data=["category", "brand_tier", "nps_score", "num_visits"], color_discrete_sequence=[THEME["gold"], THEME["blue"], THEME["green"], THEME["rose"], THEME["purple"]])
        chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        card_end()
    with c2:
        card_start("Persona profiles", "Radar view compares baskets, visits, NPS, and discount dependence at a glance.")
        st.plotly_chart(build_radar(cluster_stats), use_container_width=True)
        card_end()

    c3, c4 = st.columns(2)
    with c3:
        card_start("Target segments", "Persona summary supports the prescriptive story and action design.")
        pretty_stats = cluster_stats.rename(columns={"avg_basket": "Avg basket", "avg_discount": "Avg discount", "avg_nps": "Avg NPS", "avg_visits": "Avg visits", "flash_freq": "Flash sale freq", "premium_loyal_rate": "Premium loyal %", "revenue": "Avg revenue", "persona": "Persona", "target_action": "Recommended action"})
        st.dataframe(pretty_stats, use_container_width=True, hide_index=True, height=330)
        card_end()
    with c4:
        card_start("Association rules", "Relationship mining surfaces patterns between category, brand, channel, city, and loyalty behavior.")
        if rules.empty:
            st.info("No strong rules were found for the current filter selection.")
        else:
            fig = px.bar(rules.head(10), x="lift", y="antecedents", color="confidence", orientation="h", color_continuous_scale="Sunset")
            chart_layout(fig)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        card_end()

    card_start("Recommended actions", "These actions convert the analysis into brand and pricing decisions.")
    recs = prescriptive_recommendations(cluster_stats, rules, reg)
    r1, r2 = st.columns(2)
    for i, rec in enumerate(recs):
        with (r1 if i % 2 == 0 else r2):
            insight_box(f"Action {i+1}", rec)
    card_end()

    with st.expander("See detailed association rules"):
        st.dataframe(rules, use_container_width=True, hide_index=True)
    with st.expander("See filtered customer table"):
        cols = ["customer_id", "category", "brand_tier", "city_tier", "discount_pct", "nps_score", "avg_basket_size", "num_visits", "loyalty_label", "persona"]
        st.dataframe(clustered_df[cols].sort_values(["discount_pct", "nps_score"], ascending=[False, True]), use_container_width=True, hide_index=True, height=380)

st.download_button("Download filtered dataset", data=filtered.to_csv(index=False).encode("utf-8"), file_name="tira_customers_filtered.csv", mime="text/csv")
