import polars as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MusicDataset(Dataset):
    def __init__(self, parquet_file):
        # Load data using Polars for speed
        self.data = pl.read_parquet(parquet_file)
        
        # Define feature columns explicitly
        self.dense_cols = [
            "danceability", "energy", "speechiness", 
            "acousticness", "instrumentalness", "liveness", 
            "valence", "tempo"
        ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 1. Get Numerical Features (The Content)
        # We need to grab the row as a list, then convert to tensor
        row = self.data.row(idx, named=True)
        dense_features = [row[col] for col in self.dense_cols]
        
        # 2. Get Categorical Features (The Context)
        genre_id = row['genre_id']
        
        # 3. Return as Tensors (Float32 for weights, Long for IDs)
        return {
            "dense_features": torch.tensor(dense_features, dtype=torch.float32),
            "genre_id": torch.tensor(genre_id, dtype=torch.long)
        }

# --- TEST THE DATASET ---
dataset = MusicDataset("data/processed/songs_ready_for_model.parquet")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fetch one batch to verify shapes
batch = next(iter(loader))
print("Dense Features Shape:", batch['dense_features'].shape) # Should be [4, 8]
print("Genre ID Shape:", batch['genre_id'].shape)       # Should be [4]
print("\nSample Data:\n", batch)