### 📥 Téléchargement des données depuis Kaggle

Ce projet utilise les données de la compétition Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

#### Prérequis

* Un compte Kaggle
* La bibliothèque `kaggle` installée :

  ```bash
  pip install kaggle
  ```

#### Configuration

1. **Générer votre clé API Kaggle :**

   * Allez sur [https://www.kaggle.com/account](https://www.kaggle.com/account)
   * Cliquez sur **Create New API Token**
   * Un fichier `kaggle.json` sera téléchargé

2. **Placer le fichier `kaggle.json`** dans :

   ```
   racine\secrets\kaggle.json
   ```

#### Téléchargement automatique

Une fois la configuration faite, lancez ce script pour télécharger les données :

```python
from src.data.make_dataset import download_kaggle_data
download_kaggle_data()
```

Les fichiers seront extraits dans le dossier `data/raw/`.

