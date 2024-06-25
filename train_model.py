Script to train the recommendation model.

Model:
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import yaml

def load_data(file_path):
    return pd.read_csv(file_path)

def train_model(df, n_neighbors):
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(df)
    return model

if __name__ == "__main__":
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    df = load_data(config['data']['processed'] + "spotify_data_processed.csv")
    model = train_model(df, config['model']['parameters']['n_neighbors'])
    joblib.dump(model, config['model']['save_path'] + "music_recommender.pkl")
