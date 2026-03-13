import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_tira_dataset(n=2000):
    # --- Customer Profile ---
    customer_ids = [f"TIRA{str(i).zfill(5)}" for i in range(1, n+1)]
    ages = np.random.normal(30, 8, n).clip(18, 60).astype(int)
    genders = np.random.choice(['Female', 'Male', 'Non-binary'], n, p=[0.72, 0.23, 0.05])
    city_tiers = np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], n, p=[0.55, 0.30, 0.15])
    income_brackets = np.random.choice(
        ['₹3L–6L', '₹6L–12L', '₹12L–25L', '₹25L+'], n,
        p=[0.15, 0.35, 0.35, 0.15]
    )
    acquisition_channels = np.random.choice(
        ['App', 'In-Store', 'Influencer', 'Paid Ad', 'Word of Mouth'],
        n, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )

    # --- Purchase Behavior ---
    num_visits = np.random.poisson(8, n).clip(1, 40)
    avg_basket_size = np.random.normal(3200, 1200, n).clip(500, 12000)

    discount_pct = np.random.beta(2, 5, n) * 100  # skewed toward lower discounts
    # Tier 3 and lower income more discount-sensitive
    for i in range(n):
        if city_tiers[i] == 'Tier 3':
            discount_pct[i] = min(discount_pct[i] * 1.4, 80)
        if income_brackets[i] == '₹3L–6L':
            discount_pct[i] = min(discount_pct[i] * 1.3, 80)

    categories_bought = np.random.choice(
        ['Skincare', 'Makeup', 'Haircare', 'Fragrance', 'Luxury Tools', 'Wellness'],
        n, p=[0.35, 0.28, 0.15, 0.12, 0.05, 0.05]
    )

    brand_tier = np.random.choice(
        ['Masstige', 'Premium', 'Luxury'], n, p=[0.30, 0.45, 0.25]
    )

    # --- Discount Exposure ---
    flash_sale_response = (discount_pct > 30).astype(int)
    flash_sale_frequency = np.random.poisson(3, n).clip(0, 15)

    # --- NPS & Perception ---
    # NPS decreases as discount_pct increases (the core thesis)
    base_nps = 8 - (discount_pct / 25) + np.random.normal(0, 1, n)
    base_nps = base_nps.clip(1, 10)

    # Basket size shrinks with high discount usage over time
    basket_trend = avg_basket_size * (1 - discount_pct / 200)

    willingness_full_price = (discount_pct < 25).astype(int)
    repurchase_intent = np.where(base_nps > 7, 1, 0)

    # --- Target Variable ---
    # premium_loyal = 1 if: low discount, high NPS, high basket, premium/luxury brand
    score = (
        (discount_pct < 20) * 2 +
        (base_nps > 7) * 2 +
        (avg_basket_size > 3000) * 1 +
        (brand_tier == 'Luxury') * 2 +
        (brand_tier == 'Premium') * 1 +
        np.random.normal(0, 0.5, n)
    )
    premium_loyal = (score > 4).astype(int)

    # --- Months active ---
    months_active = np.random.randint(1, 36, n)

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'city_tier': city_tiers,
        'income_bracket': income_brackets,
        'acquisition_channel': acquisition_channels,
        'num_visits': num_visits,
        'avg_basket_size': avg_basket_size.round(0),
        'discount_pct': discount_pct.round(1),
        'category': categories_bought,
        'brand_tier': brand_tier,
        'flash_sale_response': flash_sale_response,
        'flash_sale_frequency': flash_sale_frequency,
        'nps_score': base_nps.round(1),
        'basket_trend': basket_trend.round(0),
        'willingness_full_price': willingness_full_price,
        'repurchase_intent': repurchase_intent,
        'months_active': months_active,
        'premium_loyal': premium_loyal
    })

    return df


def generate_transaction_data(df, n_transactions=5000):
    products = {
        'Skincare': ['Hydra-Boost Serum', 'Vitamin C Brightener', 'Night Repair Cream', 'SPF 50 Sunscreen'],
        'Makeup': ['Velvet Matte Lipstick', 'HD Foundation', 'Volumizing Mascara', 'Brow Definer'],
        'Haircare': ['Keratin Shampoo', 'Argan Oil Mask', 'Heat Protectant Spray'],
        'Fragrance': ['Oud Elixir EDP', 'Floral Mist EDT', 'Signature Parfum'],
        'Luxury Tools': ['Rose Quartz Roller', 'LED Face Mask', 'Sonic Cleanser'],
        'Wellness': ['Collagen Supplements', 'Vitamin Gummies', 'Herbal Face Oil']
    }

    records = []
    for _ in range(n_transactions):
        cat = np.random.choice(list(products.keys()))
        product = np.random.choice(products[cat])
        discount = np.random.beta(2, 5) * 60
        price = np.random.choice([499, 799, 1299, 1999, 2999, 4999, 7999])
        records.append({
            'product': product,
            'category': cat,
            'price': price,
            'discount_applied': round(discount, 1),
            'revenue': round(price * (1 - discount/100), 0)
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_tira_dataset()
    print(df.head())
    print(df.shape)
    print(df['premium_loyal'].value_counts())
