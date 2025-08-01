from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os
import pandas as pd
import random

app = Flask(__name__)

MODEL_PATH = 'models/best_model.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.joblib'
FEATURE_NAMES = [
    "num__Fireplaces","num__GarageArea","num__LotFrontage","num__OverallQual",
    "num__BsmtFinSF1","num__GrLivArea","num__total_bathrooms","num__WoodDeckSF",
    "num__GarageCars","num__BedroomAbvGr","num__building_age","num__BsmtUnfSF",
    "num__total_sf","num__LotArea","num__remodel_age","num__garage_age",
    "num__MasVnrArea","num__OpenPorchSF","num__MSSubClass","num__TotRmsAbvGrd",
    "cat__KitchenQual_Ex","cat__KitchenQual_Fa","cat__KitchenQual_Gd","cat__KitchenQual_TA",
    "cat__GarageType_2Types","cat__GarageType_Attchd","cat__GarageType_Basment",
    "cat__GarageType_BuiltIn","cat__GarageType_CarPort","cat__GarageType_Detchd",
    "cat__GarageType_None","cat__BsmtQual_Ex","cat__BsmtQual_Fa","cat__BsmtQual_Gd",
    "cat__BsmtQual_None","cat__BsmtQual_TA","cat__GarageFinish_Fin","cat__GarageFinish_None",
    "cat__GarageFinish_RFn","cat__GarageFinish_Unf","cat__Foundation_BrkTil",
    "cat__Foundation_CBlock","cat__Foundation_PConc","cat__Foundation_Slab",
    "cat__Foundation_Stone","cat__Foundation_Wood","cat__ExterQual_Ex",
    "cat__ExterQual_Fa","cat__ExterQual_Gd","cat__ExterQual_TA",
    "cat__Neighborhood_Blmngtn","cat__Neighborhood_Blueste","cat__Neighborhood_BrDale",
    "cat__Neighborhood_BrkSide","cat__Neighborhood_ClearCr","cat__Neighborhood_CollgCr",
    "cat__Neighborhood_Crawfor","cat__Neighborhood_Edwards","cat__Neighborhood_Gilbert",
    "cat__Neighborhood_IDOTRR","cat__Neighborhood_MeadowV","cat__Neighborhood_Mitchel",
    "cat__Neighborhood_NAmes","cat__Neighborhood_NPkVill","cat__Neighborhood_NWAmes",
    "cat__Neighborhood_NoRidge","cat__Neighborhood_NridgHt","cat__Neighborhood_OldTown",
    "cat__Neighborhood_SWISU","cat__Neighborhood_Sawyer","cat__Neighborhood_SawyerW",
    "cat__Neighborhood_Somerst","cat__Neighborhood_StoneBr","cat__Neighborhood_Timber",
    "cat__Neighborhood_Veenker"
]

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Prédiction de Prix</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(to right, #667eea, #764ba2);
            color: #333;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
        }
        .card {
            background-color: #fff;
            padding: 2rem;
            border-radius: 12px;
            max-width: 700px;
            width: 100%;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        input[type="file"] {
            margin-bottom: 1rem;
        }
        .button-group button {
            padding: 10px 20px;
            background-color: #667eea;
            border: none;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            margin: 0.5rem;
            cursor: pointer;
        }
        .result {
            margin-top: 2rem;
            background-color: #f0f0f0;
            padding: 1rem;
            border-radius: 10px;
        }
        table {
            width: 100%;
            margin-top: 1rem;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 6px 10px;
            text-align: center;
        }
    </style>
</head>
<body>
<div class="card">
    <h1>🏠 Prédiction de Prix Immobiliers</h1>
    <form method="POST" enctype="multipart/form-data">
        <p><strong>1. Charger un fichier CSV :</strong></p>
        <input type="file" name="csv_file" accept=".csv">
        <p><strong>2. Ou générer des valeurs aléatoires :</strong></p>
        <div class="button-group">
            <button type="submit" name="action" value="generate">🎲 Générer aléatoirement</button>
            <button type="submit" name="action" value="predict">📊 Prédire</button>
        </div>
    </form>
    {% if inputs %}
    <div class="result">
        <h3>🔍 Données utilisées :</h3>
        <table>
            <tr>{% for col in feature_names %}<th>{{ col }}</th>{% endfor %}</tr>
            {% for row in inputs %}
                <tr>{% for val in row %}<td>{{ val }}</td>{% endfor %}</tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    {% if predictions %}
    <div class="result">
        <h3>💰 Résultat de la prédiction :</h3>
        {% for p in predictions %}<p>➡️ {{ p }}</p>{% endfor %}
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    inputs = []
    if request.method == 'POST':
        action = request.form.get("action")

        if action == "generate":
            # Générer une ligne avec des valeurs réalistes aléatoires
            row = [round(random.uniform(-2, 3), 4) for _ in FEATURE_NAMES]
            inputs.append(row)
            if model:
                predictions = model.predict(np.array(inputs))

        elif action == "predict" and 'csv_file' in request.files:
            file = request.files['csv_file']
            if file:
                df = pd.read_csv(file)
                if set(FEATURE_NAMES).issubset(df.columns):
                    inputs = df[FEATURE_NAMES].values.tolist()
                    if model:
                        predictions = model.predict(df[FEATURE_NAMES])

    return render_template_string(HTML_TEMPLATE, predictions=predictions, inputs=inputs, feature_names=FEATURE_NAMES)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
