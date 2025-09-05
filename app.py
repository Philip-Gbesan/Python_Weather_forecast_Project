from flask import Flask, render_template, request, flash
import os
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "super-secret-key-change-this"

# Load models safely
models = {}
model_files = {
    "logistic_regression": "models/logistic_regression.pkl",
    "random_forest": "models/random_forest.pkl",
    "svm": "models/svm.pkl"
}

missing_models = []
for name, path in model_files.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
    else:
        missing_models.append(name)

# Load scaler & accuracies
scaler, accuracies = None, {}
if os.path.exists("models/scaler.pkl"):
    scaler = joblib.load("models/scaler.pkl")
if os.path.exists("models/accuracies.pkl"):
    accuracies = joblib.load("models/accuracies.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    forecast = None
    best_model = None

    if missing_models:
        flash(
            f"⚠️ Missing models: {', '.join(missing_models)}. "
            f"Run 'train_models.py' first to generate them.",
            "error"
        )
        return render_template(
            "index.html", prediction=None, accuracies=accuracies, forecast=None, best_model=None
        )

    if request.method == "POST":
        try:
            # Collect inputs
            features = [
                float(request.form.get("MinTemp", 0)),
                float(request.form.get("MaxTemp", 0)),
                float(request.form.get("Rainfall", 0)),
                float(request.form.get("WindSpeed9am", 0)),
                float(request.form.get("WindSpeed3pm", 0)),
                float(request.form.get("Humidity9am", 0)),
                float(request.form.get("Humidity3pm", 0)),
                float(request.form.get("Pressure9am", 0)),
                float(request.form.get("Pressure3pm", 0)),
                float(request.form.get("Temp9am", 0)),
                float(request.form.get("Temp3pm", 0)),
            ]
            features = np.array(features).reshape(1, -1)

            if scaler:
                features = scaler.transform(features)

            # Predictions from all models
            predictions = {}
            for name, model in models.items():
                pred = model.predict(features)[0]
                predictions[name] = "Yes" if pred == 1 else "No"

            # Majority vote forecast
            yes_votes = list(predictions.values()).count("Yes")
            no_votes = list(predictions.values()).count("No")
            forecast = "Yes" if yes_votes > no_votes else "No"

            prediction = predictions

            # Pick best model (highest accuracy from accuracies.pkl)
            if accuracies:
                best_model = max(accuracies, key=accuracies.get)

        except Exception as e:
            flash(f"❌ Error: {str(e)}", "error")

    return render_template(
        "index.html",
        prediction=prediction,
        accuracies=accuracies,
        forecast=forecast,
        best_model=best_model
    )

if __name__ == "__main__":
    app.run(debug=True)
