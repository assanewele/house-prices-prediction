from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os
import pandas as pd
import random

app = Flask(__name__)

MODEL_PATH = 'models/best_model.pkl'

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

def friendly_name(col):
    mapping = {
        "num__Fireplaces": "Nombre de cheminées",
        "num__GarageArea": "Surface du garage",
        "num__LotFrontage": "Façade du terrain",
        "num__OverallQual": "Qualité générale",
        "num__BsmtFinSF1": "Surface finie du sous-sol",
        "num__GrLivArea": "Surface habitable",
        "num__total_bathrooms": "Nombre total de salles de bain",
        "num__WoodDeckSF": "Surface terrasse bois",
        "num__GarageCars": "Nb voitures garage",
        "num__BedroomAbvGr": "Chambres à l’étage",
        "num__building_age": "Âge du bâtiment",
        "num__BsmtUnfSF": "Sous-sol non aménagé",
        "num__total_sf": "Surface totale",
        "num__LotArea": "Surface terrain",
        "num__remodel_age": "Âge rénovation",
        "num__garage_age": "Âge du garage",
        "num__MasVnrArea": "Surface maçonnerie",
        "num__OpenPorchSF": "Surface porche ouvert",
        "num__MSSubClass": "Catégorie bâtiment",
        "num__TotRmsAbvGrd": "Nb pièces (hors SDB)",
        "cat__KitchenQual_Ex": "Cuisine: Excellent",
        "cat__KitchenQual_Fa": "Cuisine: Faible",
        "cat__KitchenQual_Gd": "Cuisine: Bonne",
        "cat__KitchenQual_TA": "Cuisine: Moyenne",
        "cat__GarageType_2Types": "Garage: 2 types",
        "cat__GarageType_Attchd": "Garage: Attenant",
        "cat__GarageType_Basment": "Garage: Sous-sol",
        "cat__GarageType_BuiltIn": "Garage: Intégré",
        "cat__GarageType_CarPort": "Garage: Carport",
        "cat__GarageType_Detchd": "Garage: Détaché",
        "cat__GarageType_None": "Pas de garage",
        "cat__BsmtQual_Ex": "Sous-sol: Excellent",
        "cat__BsmtQual_Fa": "Sous-sol: Faible",
        "cat__BsmtQual_Gd": "Sous-sol: Bon",
        "cat__BsmtQual_None": "Pas de sous-sol",
        "cat__BsmtQual_TA": "Sous-sol: Moyen",
        "cat__GarageFinish_Fin": "Finition garage: Fini",
        "cat__GarageFinish_None": "Finition garage: Aucun",
        "cat__GarageFinish_RFn": "Finition garage: Moyen",
        "cat__GarageFinish_Unf": "Finition garage: Brut",
        "cat__Foundation_BrkTil": "Fondation: Brique/Tuile",
        "cat__Foundation_CBlock": "Fondation: Bloc béton",
        "cat__Foundation_PConc": "Fondation: Béton coulé",
        "cat__Foundation_Slab": "Fondation: Dalle",
        "cat__Foundation_Stone": "Fondation: Pierre",
        "cat__Foundation_Wood": "Fondation: Bois",
        "cat__ExterQual_Ex": "Extérieur: Excellent",
        "cat__ExterQual_Fa": "Extérieur: Faible",
        "cat__ExterQual_Gd": "Extérieur: Bon",
        "cat__ExterQual_TA": "Extérieur: Moyen",
        "cat__Neighborhood_Blmngtn": "Quartier: Blmngtn",
        "cat__Neighborhood_Blueste": "Quartier: Blueste",
        "cat__Neighborhood_BrDale": "Quartier: BrDale",
        "cat__Neighborhood_BrkSide": "Quartier: BrkSide",
        "cat__Neighborhood_ClearCr": "Quartier: ClearCr",
        "cat__Neighborhood_CollgCr": "Quartier: CollgCr",
        "cat__Neighborhood_Crawfor": "Quartier: Crawfor",
        "cat__Neighborhood_Edwards": "Quartier: Edwards",
        "cat__Neighborhood_Gilbert": "Quartier: Gilbert",
        "cat__Neighborhood_IDOTRR": "Quartier: IDOTRR",
        "cat__Neighborhood_MeadowV": "Quartier: MeadowV",
        "cat__Neighborhood_Mitchel": "Quartier: Mitchel",
        "cat__Neighborhood_NAmes": "Quartier: NAmes",
        "cat__Neighborhood_NPkVill": "Quartier: NPkVill",
        "cat__Neighborhood_NWAmes": "Quartier: NWAmes",
        "cat__Neighborhood_NoRidge": "Quartier: NoRidge",
        "cat__Neighborhood_NridgHt": "Quartier: NridgHt",
        "cat__Neighborhood_OldTown": "Quartier: OldTown",
        "cat__Neighborhood_SWISU": "Quartier: SWISU",
        "cat__Neighborhood_Sawyer": "Quartier: Sawyer",
        "cat__Neighborhood_SawyerW": "Quartier: SawyerW",
        "cat__Neighborhood_Somerst": "Quartier: Somerst",
        "cat__Neighborhood_StoneBr": "Quartier: StoneBr",
        "cat__Neighborhood_Timber": "Quartier: Timber",
        "cat__Neighborhood_Veenker": "Quartier: Veenker"
    }
    return mapping.get(col, col)

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
            max-height: 400px;
            overflow-y: auto;
        }
        table {
            width: 100%;
            margin-top: 1rem;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 6px 10px;
            text-align: left;
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
            <thead><tr><th>Caractéristique</th><th>Valeur</th></tr></thead>
            <tbody>
                {% for i in range(feature_names | length) %}
                    <tr>
                        <td>{{ friendly_name(feature_names[i]) }}</td>
                        <td>{{ inputs[0][i] }}</td>
                    </tr>
                {% endfor %}
            </tbody>
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

    return render_template_string(HTML_TEMPLATE, predictions=predictions, inputs=inputs, feature_names=FEATURE_NAMES, friendly_name=friendly_name)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
