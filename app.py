from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Chargement du modèle au démarrage
model_path = 'models/model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Template HTML pour le frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Prédiction du prix d'une maison 🏠</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f6f8fa;
            padding: 20px;
        }
        .container {
            background: #fff;
            padding: 30px;
            max-width: 900px;
            margin: auto;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 12px;
        }
        h1 {
            text-align: center;
            color: #444;
            margin-bottom: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
        }
        .grid input {
            padding: 8px;
            border-radius: 6px;
            border: 1px solid #ccc;
            width: 100%;
        }
        .actions {
            text-align: center;
            margin-top: 25px;
        }
        .actions button {
            background: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
        .result {
            margin-top: 30px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏡 Prédiction du prix d'une maison</h1>
        <form id="predictionForm">
            <div class="grid" id="inputGrid">
                <!-- Champs générés dynamiquement -->
            </div>
            <div class="actions">
                <button type="button" onclick="submitForm()">Faire une prédiction</button>
            </div>
        </form>
        <div class="result" id="predictionResult"></div>
    </div>

    <script>
        const features = {{ feature_names | safe }};
        const inputGrid = document.getElementById("inputGrid");

        // Générer les champs
        features.forEach((f, i) => {
            const input = document.createElement("input");
            input.type = "number";
            input.name = f;
            input.placeholder = f;
            input.id = "f" + i;
            inputGrid.appendChild(input);
        });

        async function submitForm() {
            const values = [];
            for (let i = 0; i < features.length; i++) {
                const val = document.getElementById("f" + i).value;
                if (val === "") {
                    alert("Tous les champs doivent être remplis !");
                    return;
                }
                values.push(parseFloat(val));
            }

            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ features: values })
            });

            const data = await res.json();
            if (res.ok) {
                document.getElementById("predictionResult").innerText = 
                    "🏷️ Prix estimé : " + data.prediction.toLocaleString("fr-FR") + " $";
            } else {
                document.getElementById("predictionResult").innerText = 
                    "Erreur : " + data.error;
            }
        }
    </script>
</body>
</html>
"""

# === ROUTES ===
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, feature_names=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    try:
        data = request.get_json()
        values = data.get("features", [])
        if len(values) != len(FEATURE_NAMES):
            return jsonify({"error": f"Il faut exactement {len(FEATURE_NAMES)} valeurs"}), 400
        array = np.array(values).reshape(1, -1)
        prediction = model.predict(array)
        return jsonify({"prediction": round(float(prediction[0]), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    """Informations sur le modèle"""
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    metrics_path = 'models/metrics.pkl'
    if os.path.exists(metrics_path):
        metrics = joblib.load(metrics_path)
        return jsonify(metrics)
    else:
        return jsonify({'model_type': 'RandomForestClassifier'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)