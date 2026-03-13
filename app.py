import csv
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Tira Dashboard", page_icon="📊", layout="wide")

DATA_FILE = Path("tira_customers.csv")
REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "city_tier", "income_bracket", "acquisition_channel",
    "num_visits", "avg_basket_size", "discount_pct", "category", "brand_tier",
    "flash_sale_response", "flash_sale_frequency", "nps_score", "basket_trend",
    "willingness_full_price", "repurchase_intent", "months_active", "premium_loyal"
]
NUMERIC_COLUMNS = ["num_visits", "avg_basket_size", "discount_pct", "nps_score", "months_active", "premium_loyal"]


def to_number(value):
    try:
        if value is None or value == "":
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
            clean["estimated_revenue"] = to_number(clean.get("avg_basket_size")) * to_number(clean.get("num_visits"))
            clean["loyalty_label"] = "Premium Loyal" if int(to_number(clean.get("premium_loyal"))) == 1 else "Discount-Dependent"
            rows.append(clean)
    return rows, None


def unique_values(rows, key):
    return sorted({str(r.get(key, "")) for r in rows if str(r.get(key, ""))})


def filter_rows(rows, categories, tiers, cities, max_discount):
    out = []
    for r in rows:
        if str(r.get("category")) not in categories:
            continue
        if str(r.get("brand_tier")) not in tiers:
            continue
        if str(r.get("city_tier")) not in cities:
            continue
        if to_number(r.get("discount_pct")) > max_discount:
            continue
        out.append(r)
    return out


def avg(rows, key):
    return sum(to_number(r.get(key)) for r in rows) / len(rows) if rows else 0


def summarize_by(rows, key, value_key):
    data = {}
    for r in rows:
        group = str(r.get(key, "Unknown"))
        data.setdefault(group, []).append(to_number(r.get(value_key)))
    return [{key: k, value_key: round(sum(v) / len(v), 2)} for k, v in sorted(data.items())]


def count_by_two(rows, key1, key2):
    counts = {}
    for r in rows:
        a = str(r.get(key1, "Unknown"))
        b = str(r.get(key2, "Unknown"))
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return [{key1: a, key2: b, "customers": c} for (a, b), c in sorted(counts.items())]


rows, error = load_rows()
if error:
    st.error(error)
    st.stop()

st.title("Tira Beauty Dashboard")
st.caption("Ultra-minimal version for safer deployment")

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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(filtered):,}")
c2.metric("Avg basket", f"₹{avg(filtered, 'avg_basket_size'):,.0f}")
c3.metric("Avg discount", f"{avg(filtered, 'discount_pct'):.1f}%")
c4.metric("Premium loyal", f"{avg(filtered, 'premium_loyal') * 100:.1f}%")

st.subheader("Category averages")
left, right = st.columns(2)
with left:
    st.write("Average discount by category")
    st.bar_chart(summarize_by(filtered, "category", "discount_pct"), x="category", y="discount_pct")
with right:
    st.write("Average NPS by category")
    st.bar_chart(summarize_by(filtered, "category", "nps_score"), x="category", y="nps_score")

st.subheader("Brand mix")
st.dataframe(count_by_two(filtered, "brand_tier", "loyalty_label"), use_container_width=True, hide_index=True)

st.subheader("Revenue by brand tier")
rev_data = summarize_by(filtered, "brand_tier", "estimated_revenue")
st.bar_chart(rev_data, x="brand_tier", y="estimated_revenue")

st.subheader("Customer table")
preview_cols = [
    "customer_id", "category", "brand_tier", "city_tier", "discount_pct",
    "nps_score", "avg_basket_size", "num_visits", "loyalty_label"
]
preview_rows = sorted(filtered, key=lambda r: (to_number(r.get("discount_pct")) * -1, to_number(r.get("nps_score"))))
st.dataframe([{k: r.get(k) for k in preview_cols} for r in preview_rows], use_container_width=True, hide_index=True, height=360)

csv_text = []
with DATA_FILE.open("r", encoding="utf-8-sig") as f:
    csv_text = f.read().encode("utf-8")
st.download_button("Download original CSV", data=csv_text, file_name="tira_customers.csv", mime="text/csv")
