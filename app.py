from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# === Chargement du modèle
MODEL_PATH = "models/best_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# === Liste des 75 features
FEATURE_NAMES = [
    "num__Fireplaces", "num__GarageArea", "num__LotFrontage", "num__OverallQual",
    "num__BsmtFinSF1", "num__GrLivArea", "num__total_bathrooms", "num__WoodDeckSF",
    "num__GarageCars", "num__BedroomAbvGr", "num__building_age", "num__BsmtUnfSF",
    "num__total_sf", "num__LotArea", "num__remodel_age", "num__garage_age",
    "num__MasVnrArea", "num__OpenPorchSF", "num__MSSubClass", "num__TotRmsAbvGrd",
    "cat__KitchenQual_Ex", "cat__KitchenQual_Fa", "cat__KitchenQual_Gd", "cat__KitchenQual_TA",
    "cat__GarageType_2Types", "cat__GarageType_Attchd", "cat__GarageType_Basment", "cat__GarageType_BuiltIn",
    "cat__GarageType_CarPort", "cat__GarageType_Detchd", "cat__GarageType_None",
    "cat__BsmtQual_Ex", "cat__BsmtQual_Fa", "cat__BsmtQual_Gd", "cat__BsmtQual_None", "cat__BsmtQual_TA",
    "cat__GarageFinish_Fin", "cat__GarageFinish_None", "cat__GarageFinish_RFn", "cat__GarageFinish_Unf",
    "cat__Foundation_BrkTil", "cat__Foundation_CBlock", "cat__Foundation_PConc", "cat__Foundation_Slab",
    "cat__Foundation_Stone", "cat__Foundation_Wood",
    "cat__ExterQual_Ex", "cat__ExterQual_Fa", "cat__ExterQual_Gd", "cat__ExterQual_TA",
    "cat__Neighborhood_Blmngtn", "cat__Neighborhood_Blueste", "cat__Neighborhood_BrDale",
    "cat__Neighborhood_BrkSide", "cat__Neighborhood_ClearCr", "cat__Neighborhood_CollgCr",
    "cat__Neighborhood_Crawfor", "cat__Neighborhood_Edwards", "cat__Neighborhood_Gilbert",
    "cat__Neighborhood_IDOTRR", "cat__Neighborhood_MeadowV", "cat__Neighborhood_Mitchel",
    "cat__Neighborhood_NAmes", "cat__Neighborhood_NPkVill", "cat__Neighborhood_NWAmes",
    "cat__Neighborhood_NoRidge", "cat__Neighborhood_NridgHt", "cat__Neighborhood_OldTown",
    "cat__Neighborhood_SWISU", "cat__Neighborhood_Sawyer", "cat__Neighborhood_SawyerW",
    "cat__Neighborhood_Somerst", "cat__Neighborhood_StoneBr", "cat__Neighborhood_Timber",
    "cat__Neighborhood_Veenker"
]

# === Interface utilisateur HTML intégrée
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

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

# === Lancer l'app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
