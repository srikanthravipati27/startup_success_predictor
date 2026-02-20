"""
Prosperity Prognosticator: Startup Success Prediction
Flask Web Application
"""

from flask import Flask, request, render_template
import joblib
import numpy as np
import json
import os

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'random_forest_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'feature_names.json')

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, 'r') as f:
    feature_names = json.load(f)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route('/')
def home():
    """Render the Home page."""
    return render_template('home.html')


@app.route('/predict_page')
def predict_page():
    """Render the Startup Input / Prediction form page."""
    return render_template('index.html', features=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive form values, run the model, and return the Results page.

    The form must contain float-convertible fields whose names match
    the feature list that was used during training (feature_names.json).
    """
    try:
        input_data = []
        for feat in feature_names:
            val = request.form.get(feat, 0)
            input_data.append(float(val))

        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0]

        # Map numeric prediction to label
        result_label = 'Acquired ✅' if prediction == 1 else 'Closed ❌'
        confidence = round(max(probability) * 100, 2)

        return render_template(
            'result.html',
            result=result_label,
            confidence=confidence,
            acquired_prob=round(probability[1] * 100, 2),
            closed_prob=round(probability[0] * 100, 2),
        )

    except Exception as e:
        return render_template(
            'result.html',
            result='Error',
            confidence=0,
            error_message=str(e),
            acquired_prob=0,
            closed_prob=0,
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
