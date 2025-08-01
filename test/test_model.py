import joblib
import pickle
from pathlib import Path
import unittest

# Charger les seuils
MODEL_DIR = Path("models")
metrics_path = MODEL_DIR / "xgb_best_metrics.pkl"



with open(metrics_path, "rb") as f:
    metrics = pickle.load(f)

# Afficher les métriques
RMSE_SEUIL = 0.1
MAE_SEUIL = 0.015000
R2_SEUIL = 0.85

class TestModel(unittest.TestCase):
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.metrics_path = metrics_path

    def test_model_training(self):
        """Test d'entraînement du modèle"""

        rmse = metrics.get("rmse", 0)
        mae = metrics.get("mae", 0)
        r2 = metrics.get("r", 0)

        self.assertGreater(rmse, RMSE_SEUIL)
        self.assertGreater(mae, MAE_SEUIL)   
        self.assertGreater(r2, R2_SEUIL)   

