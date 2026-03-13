import csv
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Tira Dashboard", page_icon="📊", layout="wide")

DATA_FILE = Path("tira_customers.csv")
REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "city_tier", "income_bracket", "acquisition_channel",
    "num_visits", "avg_basket_size", "discount_pct", "category", "brand_tier",
    "flash_sale_response", "flash_sale_frequency", "nps_score", "basket_trend",
    "willingness_full_price", "repurchase_intent", "months_active", "premium_loyal"
]
NUMERIC_COLUMNS = [
    "age", "num_visits", "avg_basket_size", "discount_pct", "flash_sale_response",
    "flash_sale_frequency", "nps_score", "basket_trend", "willingness_full_price",
    "repurchase_intent", "months_active", "premium_loyal"
]
PALETTE = ["#f59e0b", "#3b82f6", "#10b981", "#ec4899", "#8b5cf6", "#f97316"]

st.markdown(
    """
    <style>
        .stApp {background: linear-gradient(180deg, #09101b 0%, #0f172a 100%); color: #f8fafc;}
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px;}
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 8px 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def to_number(value):
    try:
        if value in (None, ""):
            return 0
        num = float(value)
        return int(num) if num.is_integer() else num
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def load_rows():
    if not DATA_FILE.exists():
        return None, "tira_customers.csv was not found in the same folder as app.py"
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None, "CSV file is empty or unreadable"
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            return None, "Missing columns: " + ", ".join(missing)
        rows = []
        for row in reader:
            clean = dict(row)
            for col in NUMERIC_COLUMNS:
                clean[col] = to_number(clean.get(col))
            clean["estimated_revenue"] = clean["avg_basket_size"] * clean["num_visits"]
            clean["loyalty_label"] = "Premium Loyal" if int(clean["premium_loyal"]) == 1 else "Discount-Dependent"
            clean["discount_band"] = (
                "0-10%" if clean["discount_pct"] <= 10 else
                "10-20%" if clean["discount_pct"] <= 20 else
                "20-30%" if clean["discount_pct"] <= 30 else
                "30-40%" if clean["discount_pct"] <= 40 else "40%+"
            )
            rows.append(clean)
    return rows, None


def unique_values(rows, key):
    return sorted({str(r.get(key, "")) for r in rows if str(r.get(key, ""))})


def filter_rows(rows, categories, tiers, cities, max_discount):
    return [
        r for r in rows
        if str(r.get("category")) in categories
        and str(r.get("brand_tier")) in tiers
        and str(r.get("city_tier")) in cities
        and to_number(r.get("discount_pct")) <= max_discount
    ]


def avg(rows, key):
    return sum(to_number(r.get(key)) for r in rows) / len(rows) if rows else 0


def grouped_mean(rows, group_key, value_key):
    bag = defaultdict(list)
    for r in rows:
        bag[str(r.get(group_key, "Unknown"))].append(to_number(r.get(value_key)))
    return [(k, sum(v) / len(v)) for k, v in sorted(bag.items())]


def grouped_sum(rows, group_key, value_key):
    bag = defaultdict(float)
    for r in rows:
        bag[str(r.get(group_key, "Unknown"))] += to_number(r.get(value_key))
    return [(k, v) for k, v in sorted(bag.items())]


def grouped_count(rows, key1, key2):
    bag = defaultdict(int)
    for r in rows:
        bag[(str(r.get(key1, "Unknown")), str(r.get(key2, "Unknown")))] += 1
    return [{key1: a, key2: b, "customers": c} for (a, b), c in sorted(bag.items())]


def scatter_by_tier(rows):
    tier_points = defaultdict(lambda: {"x": [], "y": [], "size": [], "text": []})
    for r in rows:
        tier = str(r.get("brand_tier", "Unknown"))
        tier_points[tier]["x"].append(to_number(r.get("discount_pct")))
        tier_points[tier]["y"].append(to_number(r.get("nps_score")))
        tier_points[tier]["size"].append(max(8, min(28, to_number(r.get("avg_basket_size")) / 180)))
        tier_points[tier]["text"].append(f"{r.get('category')} · {r.get('acquisition_channel')}")
    fig = go.Figure()
    for idx, (tier, vals) in enumerate(sorted(tier_points.items())):
        fig.add_trace(go.Scatter(
            x=vals["x"], y=vals["y"], mode="markers", name=tier,
            marker=dict(size=vals["size"], opacity=0.75, color=PALETTE[idx % len(PALETTE)], line=dict(width=0)),
            text=vals["text"], hovertemplate="Discount %{x}<br>NPS %{y}<br>%{text}<extra></extra>"
        ))
    fig.add_vline(x=25, line_dash="dash", line_color="#fbbf24")
    fig.update_layout(
        title="Discount vs NPS",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"), legend=dict(orientation="h"),
        xaxis_title="Discount %", yaxis_title="NPS score", margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def combo_discount_chart(rows):
    bands = ["0-10%", "10-20%", "20-30%", "30-40%", "40%+"]
    counts = {b: 0 for b in bands}
    nps = defaultdict(list)
    basket = defaultdict(list)
    for r in rows:
        band = r.get("discount_band")
        counts[band] += 1
        nps[band].append(to_number(r.get("nps_score")))
        basket[band].append(to_number(r.get("avg_basket_size")))
    avg_nps = [round(sum(nps[b]) / len(nps[b]), 2) if nps[b] else 0 for b in bands]
    avg_basket = [round(sum(basket[b]) / len(basket[b]), 2) if basket[b] else 0 for b in bands]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=bands, y=[counts[b] for b in bands], name="Customers", marker_color="#3b82f6"), secondary_y=False)
    fig.add_trace(go.Scatter(x=bands, y=avg_nps, name="Avg NPS", mode="lines+markers", line=dict(color="#f59e0b", width=3)), secondary_y=True)
    fig.add_trace(go.Scatter(x=bands, y=avg_basket, name="Avg basket", mode="lines+markers", line=dict(color="#10b981", width=3)), secondary_y=True)
    fig.update_layout(
        title="Discount bands performance",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"), legend=dict(orientation="h"), margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_yaxes(title_text="Customers", secondary_y=False)
    fig.update_yaxes(title_text="NPS / Basket", secondary_y=True)
    return fig


def revenue_by_category(rows):
    totals = defaultdict(float)
    for r in rows:
        key = f"{r.get('category')} · {r.get('brand_tier')}"
        totals[key] += to_number(r.get("estimated_revenue"))
    items = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:12]
    fig = go.Figure(go.Bar(
        x=[v for _, v in items],
        y=[k for k, _ in items],
        orientation="h",
        marker_color="#8b5cf6"
    ))
    fig.update_layout(
        title="Top revenue segments",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"), margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def build_pydeck(rows):
    anchors = {
        "Tier 1": {"lat": 19.0760, "lon": 72.8777, "label": "Tier 1"},
        "Tier 2": {"lat": 17.3850, "lon": 78.4867, "label": "Tier 2"},
        "Tier 3": {"lat": 26.9124, "lon": 75.7873, "label": "Tier 3"},
    }
    bucket = defaultdict(lambda: {"customers": 0, "avg_discount": 0.0, "avg_nps": 0.0})
    nps_bag = defaultdict(list)
    discount_bag = defaultdict(list)
    for r in rows:
        tier = str(r.get("city_tier", "Tier 2"))
        bucket[tier]["customers"] += 1
        nps_bag[tier].append(to_number(r.get("nps_score")))
        discount_bag[tier].append(to_number(r.get("discount_pct")))
    deck_data = []
    for tier, stats in bucket.items():
        anchor = anchors.get(tier)
        if not anchor:
            continue
        avg_nps = sum(nps_bag[tier]) / len(nps_bag[tier]) if nps_bag[tier] else 0
        avg_discount = sum(discount_bag[tier]) / len(discount_bag[tier]) if discount_bag[tier] else 0
        deck_data.append({
            "city_tier": tier,
            "label": anchor["label"],
            "lat": anchor["lat"],
            "lon": anchor["lon"],
            "customers": stats["customers"],
            "elevation": max(5000, stats["customers"] * 70),
            "avg_nps": round(avg_nps, 2),
            "avg_discount": round(avg_discount, 2),
        })
    layer = pdk.Layer(
        "ColumnLayer",
        data=deck_data,
        get_position="[lon, lat]",
        get_elevation="elevation",
        radius=32000,
        elevation_scale=1,
        pickable=True,
        auto_highlight=True,
        extruded=True,
        get_fill_color=[59, 130, 246, 180],
    )
    view = pdk.ViewState(latitude=21.5, longitude=78.5, zoom=4, pitch=45)
    tooltip = {"html": "<b>{city_tier}</b><br/>Customers: {customers}<br/>Avg discount: {avg_discount}%<br/>Avg NPS: {avg_nps}", "style": {"backgroundColor": "#111827", "color": "white"}}
    return pdk.Deck(layers=[layer], initial_view_state=view, map_style="light", tooltip=tooltip)


rows, error = load_rows()
if error:
    st.error(error)
    st.stop()

st.title("Tira Beauty Dashboard")
st.caption("Reliable CSV-based app with upgraded Plotly charts and a lightweight PyDeck view")

with st.sidebar:
    st.header("Filters")
    categories = unique_values(rows, "category")
    tiers = unique_values(rows, "brand_tier")
    cities = unique_values(rows, "city_tier")
    max_discount_all = int(max(to_number(r.get("discount_pct")) for r in rows)) if rows else 0
    selected_categories = st.multiselect("Category", categories, default=categories)
    selected_tiers = st.multiselect("Brand tier", tiers, default=tiers)
    selected_cities = st.multiselect("City tier", cities, default=cities)
    max_discount = st.slider("Max discount %", min_value=0, max_value=max_discount_all, value=max_discount_all)

filtered = filter_rows(rows, selected_categories, selected_tiers, selected_cities, max_discount)
if not filtered:
    st.warning("No data matches the selected filters.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(filtered):,}")
m2.metric("Avg basket", f"₹{avg(filtered, 'avg_basket_size'):,.0f}")
m3.metric("Avg discount", f"{avg(filtered, 'discount_pct'):.1f}%")
m4.metric("Premium loyal", f"{avg(filtered, 'premium_loyal') * 100:.1f}%")

tab1, tab2, tab3 = st.tabs(["Overview", "Map view", "Customer table"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(scatter_by_tier(filtered), use_container_width=True)
    with c2:
        st.plotly_chart(revenue_by_category(filtered), use_container_width=True)
    st.plotly_chart(combo_discount_chart(filtered), use_container_width=True)

with tab2:
    st.write("Illustrative city-tier map based on the dataset's synthetic tier segmentation.")
    st.pydeck_chart(build_pydeck(filtered), use_container_width=True)
    st.dataframe(grouped_count(filtered, "city_tier", "loyalty_label"), use_container_width=True, hide_index=True)

with tab3:
    preview_cols = [
        "customer_id", "category", "brand_tier", "city_tier", "discount_pct",
        "nps_score", "avg_basket_size", "num_visits", "loyalty_label"
    ]
    preview_rows = sorted(filtered, key=lambda r: (to_number(r.get("discount_pct")) * -1, to_number(r.get("nps_score"))))
    st.dataframe([{k: r.get(k) for k in preview_cols} for r in preview_rows], use_container_width=True, hide_index=True, height=420)

with DATA_FILE.open("r", encoding="utf-8-sig") as f:
    csv_bytes = f.read().encode("utf-8")
st.download_button("Download original CSV", data=csv_bytes, file_name="tira_customers.csv", mime="text/csv")
