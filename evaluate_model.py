Script to evaluate the trained model.

Model:
import pandas as pd
from sklearn.metrics import pairwise_distances
import joblib
import yaml

def load_data(file_path):
    return pd.read_csv(file_path)

def evaluate_model(model, df):
    distances, indices = model.kneighbors(df)
    # Add evaluation logic here
    return distances, indices

if __name__ == "__main__":
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    df = load_data(config['data']['processed'] + "spotify_data_processed.csv")
    model = joblib.load(config['model']['save_path'] + "music_recommender.pkl")
    distances, indices = evaluate_model(model, df)
    # Print or save evaluation results
