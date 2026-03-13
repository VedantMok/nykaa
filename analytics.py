import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')


def run_classification(df):
    features = ['age', 'num_visits', 'avg_basket_size', 'discount_pct',
                'flash_sale_frequency', 'nps_score', 'months_active']
    X = df[features]
    y = df['premium_loyal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    results = {
        'rf_accuracy': round(accuracy_score(y_test, rf_pred) * 100, 1),
        'lr_accuracy': round(accuracy_score(y_test, lr_pred) * 100, 1),
        'rf_auc': round(roc_auc_score(y_test, rf_proba) * 100, 1),
        'feature_importances': importances,
        'X_test': X_test,
        'y_test': y_test,
        'rf_proba': rf_proba
    }
    return results


def run_clustering(df):
    features = ['avg_basket_size', 'discount_pct', 'nps_score', 'num_visits', 'flash_sale_frequency']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df['cluster'] = labels

    persona_map = {
        df.groupby('cluster')['avg_basket_size'].mean().idxmax(): '🏆 Prestige Loyalist',
    }
    cluster_stats = df.groupby('cluster').agg({
        'avg_basket_size': 'mean',
        'discount_pct': 'mean',
        'nps_score': 'mean',
        'num_visits': 'mean',
        'flash_sale_frequency': 'mean',
        'customer_id': 'count'
    }).round(1)
    cluster_stats.columns = ['Avg Basket (₹)', 'Avg Discount %', 'Avg NPS', 'Avg Visits', 'Flash Sale Freq', 'Count']

    # Assign personas based on cluster characteristics
    persona_names = []
    for idx, row in cluster_stats.iterrows():
        if row['Avg Basket (₹)'] > cluster_stats['Avg Basket (₹)'].quantile(0.8):
            persona_names.append('🏆 Prestige Loyalist')
        elif row['Avg Discount %'] > cluster_stats['Avg Discount %'].quantile(0.8):
            persona_names.append('🎯 Deal Hunter')
        elif row['Avg NPS'] > cluster_stats['Avg NPS'].quantile(0.6) and row['Avg Basket (₹)'] > cluster_stats['Avg Basket (₹)'].median():
            persona_names.append('💄 Aspirational Buyer')
        elif row['Avg Visits'] < cluster_stats['Avg Visits'].quantile(0.3):
            persona_names.append('🌱 Explorer')
        else:
            persona_names.append('💼 Routine Professional')
    cluster_stats['Persona'] = persona_names

    return df, cluster_stats


def run_association_rules(df):
    products_by_category = {
        'Skincare': 'Skincare Products',
        'Makeup': 'Makeup Products',
        'Haircare': 'Haircare Products',
        'Fragrance': 'Fragrance',
        'Luxury Tools': 'Luxury Tools',
        'Wellness': 'Wellness'
    }

    # Create basket: each customer's categories + discount flag
    baskets = []
    for _, row in df.iterrows():
        basket = [row['category']]
        if row['discount_pct'] > 30:
            basket.append('High_Discount_Purchase')
        if row['brand_tier'] == 'Luxury':
            basket.append('Luxury_Brand')
        if row['brand_tier'] == 'Masstige':
            basket.append('Masstige_Brand')
        if row['flash_sale_response'] == 1:
            basket.append('Flash_Sale_Buyer')
        if row['repurchase_intent'] == 1:
            basket.append('High_Repurchase_Intent')
        baskets.append(basket)

    te = TransactionEncoder()
    te_array = te.fit_transform(baskets)
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_items = apriori(basket_df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_items, metric='confidence', min_threshold=0.3)
    rules = rules.sort_values('lift', ascending=False).head(15)

    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3)


def run_regression(df):
    # Linear regression: discount_pct → avg_basket_size
    X = df[['discount_pct', 'months_active', 'num_visits']].values
    y = df['avg_basket_size'].values

    lr = LinearRegression()
    lr.fit(X, y)

    discount_range = np.linspace(0, 70, 100)
    avg_months = df['months_active'].mean()
    avg_visits = df['num_visits'].mean()
    X_pred = np.column_stack([discount_range, np.full(100, avg_months), np.full(100, avg_visits)])
    basket_forecast = lr.predict(X_pred)

    # NPS regression
    X_nps = df[['discount_pct']].values
    y_nps = df['nps_score'].values
    lr_nps = LinearRegression()
    lr_nps.fit(X_nps, y_nps)
    nps_forecast = lr_nps.predict(np.linspace(0, 70, 100).reshape(-1, 1))

    return {
        'discount_range': discount_range,
        'basket_forecast': basket_forecast,
        'nps_forecast': nps_forecast,
        'basket_coef': round(lr.coef_[0], 1),
        'nps_coef': round(lr_nps.coef_[0], 3),
        'r2_basket': round(lr.score(X, y), 3)
    }
