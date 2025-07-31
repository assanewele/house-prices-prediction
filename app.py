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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps Model Prediction</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            max-width: 600px;
            width: 100%;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5rem;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1rem;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        .input-method {
            margin-bottom: 20px;
        }
        
        .input-method h3 {
            color: #555;
            margin-bottom: 15px;
            font-size: 1.2rem;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .feature-input {
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            text-align: center;
            transition: border-color 0.3s;
        }
        
        .feature-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .textarea-input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            min-height: 80px;
            font-family: monospace;
        }
        
        .textarea-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin: 10px 0;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            display: none;
        }
        
        .result.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .result.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .examples {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .examples h4 {
            color: #555;
            margin-bottom: 10px;
        }
        
        .example-btn {
            background: #6c757d;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            margin: 2px;
            font-size: 12px;
        }
        
        .example-btn:hover {
            background: #5a6268;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 ML Model Predictor</h1>
        <p class="subtitle">Entrez 20 valeurs numériques pour obtenir une prédiction</p>
        
        <div class="input-section">
            <div class="input-method">
                <h3>📊 Saisie par grille (20 features)</h3>
                <div class="features-grid" id="featuresGrid">
                    <!-- Les inputs seront générés par JavaScript -->
                </div>
                <button type="button" class="btn" onclick="generateRandomValues()">🎲 Valeurs aléatoires</button>
            </div>
            
            <div class="input-method">
                <h3>📝 Saisie rapide (format JSON)</h3>
                <textarea class="textarea-input" id="jsonInput" 
                         placeholder='Exemple: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'></textarea>
                <button type="button" class="btn" onclick="loadFromJson()">📥 Charger depuis JSON</button>
            </div>
        </div>
        
        <button type="button" class="btn" onclick="makePrediction()" id="predictBtn">
            🚀 Faire une prédiction
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Prédiction en cours...</p>
        </div>
        
        <div class="result" id="result"></div>
        
        <div class="examples">
            <h4>💡 Exemples rapides :</h4>
            <button class="example-btn" onclick="loadExample1()">Exemple 1</button>
            <button class="example-btn" onclick="loadExample2()">Exemple 2</button>
            <button class="example-btn" onclick="loadExample3()">Exemple 3</button>
        </div>
    </div>

    <script>
        // Générer la grille d'inputs
        function initializeGrid() {
            const grid = document.getElementById('featuresGrid');
            for (let i = 0; i < 20; i++) {
                const input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.className = 'feature-input';
                input.placeholder = `F${i+1}`;
                input.id = `feature${i}`;
                grid.appendChild(input);
            }
        }
        
        // Générer des valeurs aléatoires
        function generateRandomValues() {
            for (let i = 0; i < 20; i++) {
                const input = document.getElementById(`feature${i}`);
                input.value = (Math.random() * 10 - 5).toFixed(2);
            }
        }
        
        // Charger depuis JSON
        function loadFromJson() {
            try {
                const jsonText = document.getElementById('jsonInput').value;
                if (!jsonText.trim()) return;
                
                const values = JSON.parse(jsonText);
                if (values.length !== 20) {
                    alert('Le JSON doit contenir exactement 20 valeurs');
                    return;
                }
                
                for (let i = 0; i < 20; i++) {
                    document.getElementById(`feature${i}`).value = values[i];
                }
                
                showResult('✅ Valeurs chargées depuis JSON', 'success');
            } catch (error) {
                showResult('❌ Format JSON invalide', 'error');
            }
        }
        
        // Exemples prédéfinis
        function loadExample1() {
            const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
            loadValues(values);
        }
        
        function loadExample2() {
            const values = [-1, -2, 3, 4, -5, 6, -7, 8, 9, -10, 11, -12, 13, 14, -15, 16, -17, 18, 19, -20];
            loadValues(values);
        }
        
        function loadExample3() {
            const values = [0.5, 1.2, -0.8, 2.1, 0.9, -1.5, 0.3, 1.8, -0.4, 2.5, 0.7, -1.2, 0.6, 1.9, -0.3, 2.2, 0.8, -1.1, 0.4, 1.7];
            loadValues(values);
        }
        
        function loadValues(values) {
            for (let i = 0; i < 20; i++) {
                document.getElementById(`feature${i}`).value = values[i];
            }
            showResult('✅ Exemple chargé', 'success');
        }
        
        // Faire une prédiction
        async function makePrediction() {
            const features = [];
            
            // Récupérer les valeurs
            for (let i = 0; i < 20; i++) {
                const value = document.getElementById(`feature${i}`).value;
                if (value === '') {
                    showResult('❌ Veuillez remplir toutes les 20 features', 'error');
                    return;
                }
                features.push(parseFloat(value));
            }
            
            // Afficher le loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('predictBtn').disabled = true;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ features: features })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    const prediction = data.prediction;
                    const probability = data.probability;
                    const confidence = Math.max(...probability) * 100;
                    
                    showResult(
                        `🎯 Prédiction: <strong>Classe ${prediction}</strong><br>
                         📊 Confiance: <strong>${confidence.toFixed(1)}%</strong><br>
                         📈 Probabilités: [${probability.map(p => p.toFixed(3)).join(', ')}]`,
                        'success'
                    );
                } else {
                    showResult(`❌ Erreur: ${data.error}`, 'error');
                }
            } catch (error) {
                showResult(`❌ Erreur de connexion: ${error.message}`, 'error');
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('predictBtn').disabled = false;
            }
        }
        
        // Afficher les résultats
        function showResult(message, type) {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = message;
            resultDiv.className = `result ${type}`;
            resultDiv.style.display = 'block';
            
            // Auto-hide success messages after 3 seconds
            if (type === 'success' && !message.includes('Prédiction:')) {
                setTimeout(() => {
                    resultDiv.style.display = 'none';
                }, 3000);
            }
        }
        
        // Initialiser au chargement
        document.addEventListener('DOMContentLoaded', initializeGrid);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Page d'accueil avec interface utilisateur"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction (API)"""
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

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