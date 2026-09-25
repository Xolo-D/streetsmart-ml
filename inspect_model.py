import joblib
import sys

try:
    m = joblib.load('model.pkl')
except FileNotFoundError:
    sys.exit("❌ model.pkl not found in this folder")
except Exception as e:
    sys.exit(f"❌ Could not load model: {e}")

print("=== MODEL INFO ===")
print("Type:", type(m).__name__)
print("Module:", type(m).__module__)
print()

if hasattr(m, 'feature_names_in_'):
    print("FEATURES THE MODEL EXPECTS:")
    for f in m.feature_names_in_:
        print("  -", f)
    print()
    print("Total features:", len(m.feature_names_in_))
elif hasattr(m, 'n_features_in_'):
    print("N features:", m.n_features_in_)
else:
    print("No feature metadata available")

print()

# Extra: try to get the training target name
if hasattr(m, 'get_booster'):
    try:
        booster = m.get_booster()
        print("Booster feature names:", booster.feature_names)
    except Exception:
        pass

print()
print("✅ Model loaded successfully")
