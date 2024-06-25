Script for data cleaning, preprocessing, and transformation.

Model:
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    return pd.DataFrame(df_scaled, columns=df.select_dtypes(include=['float64', 'int64']).columns)

if __name__ == "__main__":
    df = load_data("data/raw/spotify_data.csv")
    df_processed = preprocess_data(df)
    df_processed.to_csv("data/processed/spotify_data_processed.csv", index=False)
