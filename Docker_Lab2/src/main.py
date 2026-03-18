import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model artifacts
with open("wine_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]
class_names = artifacts["class_names"]

FEATURE_LABELS = [
    ("alcohol", "Alcohol", "% vol"),
    ("malic_acid", "Malic Acid", "g/L"),
    ("ash", "Ash", "g/L"),
    ("alcalinity_of_ash", "Alcalinity of Ash", "mEq/L"),
    ("magnesium", "Magnesium", "mg/L"),
    ("total_phenols", "Total Phenols", "g/L"),
    ("flavanoids", "Flavanoids", "g/L"),
    ("nonflavanoid_phenols", "Nonflavanoid Phenols", "g/L"),
    ("proanthocyanins", "Proanthocyanins", "g/L"),
    ("color_intensity", "Color Intensity", ""),
    ("hue", "Hue", ""),
    ("od280_od315_of_diluted_wines", "OD280/OD315", ""),
    ("proline", "Proline", "mg/L"),
]

EXAMPLE_VALUES = [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.40, 1050.0]


@app.route("/")
def index():
    return render_template(
        "predict.html",
        features=FEATURE_LABELS,
        examples=EXAMPLE_VALUES,
        class_names=list(class_names),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form.get(key, 0)) for key, _, _ in FEATURE_LABELS]
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred_idx = int(model.predict(X_scaled)[0])
        proba = model.predict_proba(X_scaled)[0].tolist()
        return jsonify({
            "prediction": class_names[pred_idx],
            "probabilities": {class_names[i]: round(p * 100, 1) for i, p in enumerate(proba)},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=False)
