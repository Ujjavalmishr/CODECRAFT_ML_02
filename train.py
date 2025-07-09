# train.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from joblib import dump
from typing import List, Tuple

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.replace('–', '-', regex=False)  # Normalize column names
    return df

def preprocess_data(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=features), scaler

def train_kmeans(X_scaled: pd.DataFrame, k: int) -> Tuple[pd.Series, KMeans]:
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(X_scaled)
    return pd.Series(clusters), model

def get_centroids(model: KMeans, scaler: StandardScaler, features: List[str]) -> pd.DataFrame:
    centroids_scaled = model.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    return pd.DataFrame(centroids, columns=features)

def save_model(model: KMeans, filename: str = 'model/kmeans_model.pkl') -> None:
    dump(model, filename)
