import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def create_features(df):
    """Créer de nouvelles features à partir des données existantes.

    Args:
        df (pd.DataFrame): DataFrame contenant les données brutes.

    Returns:
        pd.DataFrame: DataFrame avec les nouvelles features.
    """
    df = df.copy()
    df["building_age"] = df["YrSold"].astype(float) - df["YearBuilt"].astype(float)
    df["remodel_age"] = df["YrSold"].astype(float) - df["YearRemodAdd"].astype(float)
    df["garage_age"] = df["YrSold"].astype(float) - df["GarageYrBlt"].fillna(0).astype(float)
    df["total_sf"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"].fillna(0)
    df["total_bathrooms"] = (
        df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    return df


def impute_missing_values(df, categorical_cols, numeric_cols):
    """Imputer les valeurs manquantes dans les colonnes spécifiées.

    Args:
        df (pd.DataFrame): DataFrame à traiter.
        categorical_cols (list): Liste des colonnes catégoriques.
        numeric_cols (list): Liste des colonnes numériques.

    Returns:
        pd.DataFrame: DataFrame avec valeurs manquantes imputées.
    """
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_outliers(df, numeric_cols):
    """Corriger les outliers dans les colonnes numériques avec la méthode IQR.

    Args:
        df (pd.DataFrame): DataFrame à traiter.
        numeric_cols (list): Liste des colonnes numériques.

    Returns:
        pd.DataFrame: DataFrame avec outliers capés.
    """
    df = df.copy()
    for feature in numeric_cols:
        if feature in df.columns:
            q1 = df[feature].quantile(0.25)
            q3 = df[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[feature] = df[feature].apply(
                lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x
            )
    return df


def preprocess_data(train, test, categorical_cols, numeric_cols):
    """Prétraiter les données avec encodage et standardisation.

    Args:
        train (pd.DataFrame): DataFrame d'entraînement.
        test (pd.DataFrame): DataFrame de test.
        categorical_cols (list): Colonnes catégoriques.
        numeric_cols (list): Colonnes numériques.

    Returns:
        tuple: (X_train_processed, X_test_processed, preprocessor)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    X_train_processed = preprocessor.fit_transform(train[numeric_cols + categorical_cols])
    X_test_processed = preprocessor.transform(test[numeric_cols + categorical_cols])
    return X_train_processed, X_test_processed, preprocessor