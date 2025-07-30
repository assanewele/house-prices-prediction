import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def impute_missing_values(df, categorical_cols, numeric_cols):
    """Imputer les valeurs manquantes : 'None' pour les catégoriques, médiane pour les numériques."""
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df

def create_features(df):
    """Créer des features : HouseAge, TotalSF, TotalBathrooms, OverallQualityCond."""
    df = df.copy()
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['OverallQualityCond'] = df['OverallQual'] * df['OverallCond']
    return df

def remove_non_pertinent_cols(df, cols_to_drop):
    """Supprimer les colonnes non pertinentes (valeurs manquantes élevées, faible corrélation, redondantes)."""
    df = df.copy()
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    return df.drop(cols_to_drop, axis=1)

def remove_outliers(df, outlier_thresholds=None):
    """Supprimer les outliers pour les variables spécifiées."""
    df = df.copy()
    if outlier_thresholds is None:
        outlier_thresholds = {
            'OverallQual': 10,
            'GrLivArea': 4000,
            'TotalSF': 6000,
            'GarageCars': 3,
            'TotalBathrooms': 5
        }
    for col, threshold in outlier_thresholds.items():
        if col in df.columns:
            df = df[df[col] <= threshold]
    return df

def encode_and_scale(train, test, categorical_cols, numeric_cols):
    """Encoder les variables catégoriques (One-Hot) et standardiser les variables numériques."""
    train = train.copy()
    test = test.copy()
    train = pd.get_dummies(train, columns=[col for col in categorical_cols if col in train.columns], drop_first=True)
    test = pd.get_dummies(test, columns=[col for col in categorical_cols if col in test.columns], drop_first=True)
    train, test = train.align(test, join='left', axis=1, fill_value=0)
    scaler = StandardScaler()
    numeric_cols = [col for col in numeric_cols if col in train.columns and col in test.columns]
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])
    return train, test, scaler