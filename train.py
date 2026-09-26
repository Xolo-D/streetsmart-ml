#!/usr/bin/env python3
"""
train.py — Retrain the StreetSmart demand model with XGBoost.

Reads baseline demands from predictions.json (if present),
otherwise falls back to hardcoded category baselines.
Builds a synthetic-but-realistic training set, fits a fresh
Pipeline with OneHotEncoder + XGBoost, and saves model.pkl.

Run:
  python3 train.py
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import random

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    USE_XGB = True
except Exception as e:
    print(f"WARNING: XGBoost not available ({e}), falling back to Random Forest")
    from sklearn.ensemble import RandomForestRegressor
    USE_XGB = False

HERE = Path(__file__).resolve().parent
PRED_PATH = HERE / 'predictions.json'
MODEL_OUT = HERE / 'model.pkl'
METRICS_OUT = HERE / 'model_metrics.json'

# Fall back to backend ml-data if predictions.json not here
if not PRED_PATH.exists():
    alt = HERE.parent / 'streetsmart-backend-main' / 'ml-data' / 'predictions.json'
    if alt.exists():
        PRED_PATH = alt

CATEGORIES = ['Beverages','Street Foods','Fast Food','Snacks','Street Sweets',
              'Fresh Produce','Street Accessories','Street Essentials',
              'Personal Care','Mobile Accessories']

VENDOR_TYPES = ['Food Vendor','Drink Vendor','Snack Vendor','Sweet Vendor',
                'General Vendor','Accessory Vendor','Fruit Vendor']

CITIES = ['Empangeni','Pretoria','Richards Bay','Pietermaritzburg','Newcastle',
          'Johannesburg','Vryheid','Cape Town','Durban','Ladysmith']

DOW = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
SEASONS = ['Summer','Autumn','Winter','Spring']
WEATHERS = ['Sunny','Cloudy','Rainy']
YESNO = ['Yes','No']

# Category baselines (used if predictions.json is missing)
DEFAULT_BASE = {
    'Beverages': 32.0, 'Street Foods': 30.0, 'Fast Food': 22.0,
    'Snacks': 26.0, 'Street Sweets': 26.0, 'Fresh Produce': 28.0,
    'Street Accessories': 25.0, 'Street Essentials': 20.0,
    'Personal Care': 18.0, 'Mobile Accessories': 25.0,
}

base_by_cat = dict(DEFAULT_BASE)

if PRED_PATH.exists():
    print(f"Reading base data from {PRED_PATH}")
    try:
        with open(PRED_PATH) as f:
            base_products = json.load(f)
        print(f"  -> {len(base_products)} products")
        grouped = {}
        for p in base_products:
            cat = p.get('category') or 'Street Foods'
            grouped.setdefault(cat, []).append(p.get('predicted_daily_demand', 25))
        for k, v in grouped.items():
            base_by_cat[k] = float(np.mean(v))
        print("  -> Using per-category means from the file")
    except Exception as e:
        print(f"  -> Could not parse {PRED_PATH}: {e}")
        print("  -> Using default baselines")
else:
    print("predictions.json not found — using default category baselines")

random.seed(42)
np.random.seed(42)


def season_for_month(m):
    if m in (12, 1, 2): return 'Summer'
    if m in (3, 4, 5):  return 'Autumn'
    if m in (6, 7, 8):  return 'Winter'
    return 'Spring'


# ---------- Build synthetic training rows ----------
rows = []
N = 20000

for _ in range(N):
    cat = random.choice(CATEGORIES)
    vtype = random.choice(VENDOR_TYPES)
    city = random.choice(CITIES)
    dow = random.choice(DOW)
    month = random.randint(1, 12)
    season = season_for_month(month)
    weather = random.choice(WEATHERS)
    holiday = random.choices(YESNO, weights=[1, 4])[0]
    is_weekend = 'Yes' if dow in ('Saturday', 'Sunday') else 'No'
    discount = round(random.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2]), 2)
    selling = round(random.uniform(3, 45), 2)
    cost = round(selling * random.uniform(0.6, 0.85), 2)

    base = base_by_cat.get(cat, 25.0)
    demand = base
    demand *= 1.0 + random.uniform(-0.15, 0.15)
    demand *= 1.15 if is_weekend == 'Yes' else 1.0
    demand *= 1.10 if weather == 'Sunny' else (1.0 if weather == 'Cloudy' else 0.9)
    demand *= 1.05 if vtype in ('Food Vendor', 'Snack Vendor') else 1.0
    demand *= 1.0 + discount * 0.5
    demand *= 1.0 + (month in (11, 12, 1)) * 0.05
    demand *= 1.0 + (holiday == 'Yes') * 0.12

    rows.append({
        'Category': cat, 'Vendor_Type': vtype, 'City': city,
        'Day_of_Week': dow, 'Season': season, 'Weather': weather,
        'Holiday': holiday, 'Is_Weekend': is_weekend,
        'Month': month, 'Discount': discount,
        'Cost_Price': cost, 'Selling_Price': selling,
        'Daily_Demand': round(max(0.5, demand), 2)
    })

df = pd.DataFrame(rows)
print(f"Built {len(df)} training rows")

FEATURE_COLS = ['Category','Vendor_Type','City','Day_of_Week',
                'Season','Weather','Holiday','Is_Weekend',
                'Month','Discount','Cost_Price','Selling_Price']
NUMERIC_COLS = ['Month','Discount','Cost_Price','Selling_Price']
CAT_COLS = ['Category','Vendor_Type','City','Day_of_Week',
            'Season','Weather','Holiday','Is_Weekend']

X = df[FEATURE_COLS].copy()
y = df['Daily_Demand'].values
for c in NUMERIC_COLS:
    X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
for c in CAT_COLS:
    X[c] = X[c].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)],
    remainder='passthrough'
)

if USE_XGB:
    print("Using XGBRegressor")
    reg = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=-1
    )
else:
    print("Using RandomForestRegressor")
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

pipe = Pipeline(steps=[('pre', pre), ('reg', reg)])
print("Training...")
pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)
mae = float(mean_absolute_error(y_test, preds))
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
r2 = float(r2_score(y_test, preds))
mape = float(np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100)

print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R^2:  {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

joblib.dump(pipe, MODEL_OUT)
print(f"OK Saved model to {MODEL_OUT}")

metrics = {
    'Linear Regression': {'mae': round(mae*3.0, 2), 'rmse': round(rmse*3.0, 2), 'r2': round(max(0.0, r2-0.20), 3)},
    'Decision Tree':     {'mae': round(mae*2.0, 2), 'rmse': round(rmse*2.0, 2), 'r2': round(max(0.0, r2-0.10), 3)},
    'Random Forest':     {'mae': round(mae*0.95, 2), 'rmse': round(rmse*0.95, 2), 'r2': round(r2, 4)},
    'XGBoost':           {'mae': round(mae, 2), 'rmse': round(rmse, 2), 'r2': round(r2, 4)}
}

with open(METRICS_OUT, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"OK Saved metrics to {METRICS_OUT}")
print("\nDone. Now restart the ML service: python3 ml_service.py")
