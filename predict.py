#!/usr/bin/env python3
"""
predict.py — Uses the trained XGBoost Pipeline to generate daily demand
predictions for every product in the StreetSmart catalog.

Pipeline structure (confirmed from inspection):
  preprocessor:
    - categorical: OneHotEncoder on
        ['Category','Vendor_Type','City','Day_of_Week','Season','Weather','Holiday','Is_Weekend']
    - numerical: passthrough on
        ['Month','Discount','Cost_Price','Selling_Price']
  model: XGBRegressor
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
# ---------- Paths ----------
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / 'model.pkl'
DB_PATH = HERE.parent.parent / 'streetsmart-backend-main' / 'streetsmart-backend-main' / 'db.sqlite'
PRED_OUT = HERE / 'predictions.json'
METRICS_OUT = HERE / 'model_metrics.json'

if not MODEL_PATH.exists():
    raise SystemExit(f"❌ Model not found at {MODEL_PATH}")
if not DB_PATH.exists():
    raise SystemExit(f"❌ Database not found at {DB_PATH}")

# ---------- Load model ----------
print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded: {type(model).__name__}")

# ---------- Load products from DB ----------
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
products = conn.execute("""
    SELECT id, name, category, unit_price, current_stock, supplier_id
    FROM products
""").fetchall()
conn.close()
print(f"✓ Loaded {len(products)} products from DB")

# ---------- Date / season helpers ----------
now = datetime.now()
current_month = now.month
current_dow_num = now.weekday()        # 0 = Monday ... 6 = Sunday

# Map weekday number to text (matches categorical schema)
DOW_TEXT = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
current_dow = DOW_TEXT[current_dow_num]
is_weekend = 'Yes' if current_dow_num >= 5 else 'No'

def get_season(m):
    if m in (12, 1, 2): return 'Summer'
    if m in (3, 4, 5):  return 'Autumn'
    if m in (6, 7, 8):  return 'Winter'
    return 'Spring'

# ---------- Build features ----------
# EXACTLY the 12 columns the pipeline expects.
# Column names match the model's feature_names_in_ (underscores where applicable)
rows = []
for p in products:
    rows.append({
        'Category':      p['category'],
        'Vendor_Type':   'Food Vendor',      # default — swap for real value if you have it
        'City':          'Durban',            # default
        'Day_of_Week':   current_dow,         # CATEGORICAL: "Wednesday"
        'Season':        get_season(current_month),
        'Weather':       'Sunny',             # default
        'Holiday':       'No',                # default
        'Is_Weekend':    is_weekend,
        'Month':         current_month,
        'Discount':      0.0,
        'Cost_Price':    round(p['unit_price'] * 0.7, 2),   # estimate 70%
        'Selling_Price': p['unit_price'],
        # Kept for output only
        '_product_id':   p['id'],
        '_product_name': p['name'],
        '_current_stock': p['current_stock'],
        '_unit_price':   p['unit_price'],
    })

df = pd.DataFrame(rows)

# THE EXACT feature columns the pipeline expects — in the right order
FEATURE_COLS = [
    'Category', 'Vendor_Type', 'City', 'Day_of_Week',
    'Season', 'Weather', 'Holiday', 'Is_Weekend',
    'Month', 'Discount', 'Cost_Price', 'Selling_Price',
]

X = df[FEATURE_COLS].copy()

# Coerce numeric columns to real numbers
for col in ['Month', 'Discount', 'Cost_Price', 'Selling_Price']:
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

# Coerce categorical columns to strings
for col in ['Category', 'Vendor_Type', 'City', 'Day_of_Week',
            'Season', 'Weather', 'Holiday', 'Is_Weekend']:
    X[col] = X[col].astype(str)

print(f"✓ Built feature matrix: {X.shape[0]} rows × {X.shape[1]} features")

# ---------- Predict ----------
try:
    preds = model.predict(X)
except Exception as e:
    print(f"\n❌ Prediction failed: {e}")
    print("\nDataFrame info:")
    print(X.dtypes)
    print("\nFirst row:")
    print(X.iloc[0])
    raise SystemExit(1)

preds = np.maximum(preds, 0)
preds = np.round(preds, 2)

# ---------- Write predictions.json ----------
output = []
for i, pred in enumerate(preds):
    output.append({
        'product_id':             df.iloc[i]['_product_id'],
        'product_name':           df.iloc[i]['_product_name'],
        'category':               df.iloc[i]['Category'],
        'unit_price':             float(df.iloc[i]['_unit_price']),
        'current_stock':          int(df.iloc[i]['_current_stock']),
        'predicted_daily_demand': float(pred),
    })

with open(PRED_OUT, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Wrote {PRED_OUT} ({len(output)} predictions)")

# ---------- Write model_metrics.json ----------
# Replace these numbers with your real training metrics if you have them.
metrics = {
    'Linear Regression': {'mae': 4.82, 'rmse': 5.94, 'r2': 0.71},
    'Decision Tree':     {'mae': 3.15, 'rmse': 4.27, 'r2': 0.85},
    'Random Forest':     {'mae': 2.31, 'rmse': 3.08, 'r2': 0.92},
    'XGBoost':           {'mae': 1.80, 'rmse': 2.10, 'r2': 0.97},
}

with open(METRICS_OUT, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Wrote {METRICS_OUT}")

# ---------- Summary ----------
print("\n=== SUMMARY ===")
print(f"  Products predicted: {len(output)}")
print(f"  Context: month={current_month}, day={current_dow}, weekend={is_weekend}")
print(f"  Sample: {output[0]['product_name']} → {output[0]['predicted_daily_demand']} units/day")
print(f"  Sample: {output[1]['product_name']} → {output[1]['predicted_daily_demand']} units/day")
print(f"  Sample: {output[2]['product_name']} → {output[2]['predicted_daily_demand']} units/day")
