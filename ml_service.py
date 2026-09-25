#!/usr/bin/env python3
"""
ml_service.py — Live Flask inference service for StreetSmart.

Loads model.pkl once at startup and exposes:
  POST /predict          -> single prediction
  POST /predict/batch    -> batch predictions
  GET  /health           -> health check
  GET  /features         -> list of expected features (debug)

Run:
  python ml_service.py
Runs on http://localhost:5001
"""

# ============================================================
# COMPATIBILITY PATCH — must run before sklearn is used
# ============================================================
try:
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    def _patched_get_tags(self):
        try:
            if hasattr(self, '_get_tags'):
                return self._get_tags()
        except Exception:
            pass
        return {
            'allow_nan': True,
            'requires_y': False,
            'requires_fit': True,
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'binary_only': False,
            'pairwise': False,
            'preserves_dtype': [],
            'X_types': ['2darray'],
            'input_tags': {'sparse': False, 'categorical': False, 'one_d_array': False, 'two_d_array': True, 'three_d_array': False},
            'target_tags': {'required': False, 'one_d_array': True, 'two_d_array': False, 'multi_output': False, 'single_output': True},
            'transformer_tags': {'allow_nan': True, 'preserves_dtype': [], 'sparse_output': False}
        }

    for cls in [BaseEstimator, Pipeline, ColumnTransformer, OneHotEncoder]:
        if not hasattr(cls, '__sklearn_tags__'):
            cls.__sklearn_tags__ = _patched_get_tags
except Exception as _e:
    print(f"[warn] Compat patch skipped: {_e}")
# ============================================================

import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / 'model.pkl'

if not MODEL_PATH.exists():
    raise SystemExit(f"ERROR: Model not found at {MODEL_PATH}")

print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print(f"OK Model loaded: {type(model).__name__}")

FEATURE_COLS = [
    'Category', 'Vendor_Type', 'City', 'Day_of_Week',
    'Season', 'Weather', 'Holiday', 'Is_Weekend',
    'Month', 'Discount', 'Cost_Price', 'Selling_Price'
]

NUMERIC_COLS = ['Month', 'Discount', 'Cost_Price', 'Selling_Price']
CATEGORICAL_COLS = ['Category', 'Vendor_Type', 'City', 'Day_of_Week',
                    'Season', 'Weather', 'Holiday', 'Is_Weekend']

app = Flask(__name__)
CORS(app)


def build_dataframe(rows):
    df = pd.DataFrame(rows)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = '' if col in CATEGORICAL_COLS else 0
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str)
    return df[FEATURE_COLS]


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': type(model).__name__})


@app.route('/features', methods=['GET'])
def features():
    return jsonify({
        'features': FEATURE_COLS,
        'numeric': NUMERIC_COLS,
        'categorical': CATEGORICAL_COLS
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        df = build_dataframe([data])
        prediction = model.predict(df)[0]
        prediction = max(0, float(prediction))
        prediction = round(prediction, 2)

        return jsonify({
            'predicted_daily_demand': prediction,
            'input': data
        })
    except Exception as e:
        print("\n========== PREDICT ERROR ==========")
        traceback.print_exc()
        print("===================================\n")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json(force=True)
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected a JSON array'}), 400

        df = build_dataframe(data)
        preds = model.predict(df)
        preds = np.maximum(preds, 0)
        preds = np.round(preds, 2)

        results = []
        for i, pred in enumerate(preds):
            results.append({
                'predicted_daily_demand': float(pred),
                'input': data[i]
            })
        return jsonify({'results': results})
    except Exception as e:
        print("\n========== BATCH ERROR ==========")
        traceback.print_exc()
        print("=================================\n")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("StreetSmart ML service at http://localhost:5001")
    print("   Test:    http://localhost:5001/health")
    print("   Predict: POST http://localhost:5001/predict\n")
    app.run(host='0.0.0.0', port=5001, debug=False)