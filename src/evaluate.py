import torch
import polars as pl
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from model import TwoTowerModel

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/processed/songs_final.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "two_tower_model.pth")
ENCODER_PATH = os.path.join(BASE_DIR, "../data/processed/mood_encoder.pkl")
DEVICE = torch.device("cpu") # Inference is fast enough on CPU

def load_artifacts():
    # 1. Load Data
    df = pl.read_parquet(DATA_PATH)
    
    # 2. Load Encoders
    with open(ENCODER_PATH, "rb") as f:
        mood_encoder = pickle.load(f)
        
    # 3. Load Model
    num_genres = df["genre_id"].max() + 1
    num_moods = len(mood_encoder.classes_)
    
    model = TwoTowerModel(
        audio_feature_dim=8, # standard dense cols
        num_genres=num_genres,
        num_moods=num_moods
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    return df, model, mood_encoder

def generate_song_embeddings(df, model):
    """
    Passes ALL songs through the Item Tower to create the 'Vector Database'.
    """
    print("Indexing songs... (Generating Embeddings)")
    dense_cols = [
        "danceability", "energy", "speechiness", 
        "acousticness", "instrumentalness", "liveness", 
        "valence", "tempo"
    ]
    
    # Convert to Tensor
    dense_feats = torch.tensor(df.select(dense_cols).to_numpy(), dtype=torch.float32)
    genre_ids = torch.tensor(df["genre_id"].to_numpy(), dtype=torch.long)
    
    with torch.no_grad():
        # Only use the Item Tower!
        song_vectors = model.item_tower(dense_feats, genre_ids)
        
    return song_vectors.numpy()

def evaluate_hit_rate(df, song_vectors, model, mood_encoder, k=10):
    """
    Business Metric: HR@10
    For each mood, we query the system. 
    If the Top 10 results match the requested mood, it's a 'Hit'.
    """
    print(f"\n--- Evaluation: Hit Rate @ {k} ---")
    
    moods = mood_encoder.classes_
    
    for mood_name in moods:
        # 1. Create a Query Vector for this mood
        mood_idx = mood_encoder.transform([mood_name])[0]
        mood_tensor = torch.tensor([mood_idx], dtype=torch.long)
        
        with torch.no_grad():
            query_vec = model.query_tower(mood_tensor).numpy()
            
        # 2. Similarity Search (Dot Product)
        # (1, 64) x (N, 64).T = (1, N) scores
        scores = cosine_similarity(query_vec, song_vectors)
        
        # 3. Get Top K Indices
        top_k_indices = np.argsort(scores[0])[::-1][:k]
        
        # 4. Check Ground Truth
        retrieved_rows = df[top_k_indices]
        retrieved_moods = retrieved_rows["mood_label"].to_list()
        
        # 5. Calculate Accuracy for this Query
        hits = sum([1 for m in retrieved_moods if m == mood_name])
        accuracy = hits / k
        
        print(f"Query: '{mood_name}' -> Top {k} Accuracy: {accuracy:.0%} | retrieved: {retrieved_moods[:3]}...")

if __name__ == "__main__":
    df, model, encoder = load_artifacts()
    song_vectors = generate_song_embeddings(df, model)
    evaluate_hit_rate(df, song_vectors, model, encoder)