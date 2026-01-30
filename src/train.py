import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import polars as pl
import numpy as np
import os

# Import your custom model
from model import TwoTowerModel 

# --- CONFIG ---
# Use absolute path to avoid "FileNotFound" errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/processed/songs_final.parquet")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "two_tower_model.pth")

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# Detect Device (GPU is much faster)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

class MusicDataset(Dataset):
    def __init__(self, parquet_file):
        # 1. Load Data
        df = pl.read_parquet(parquet_file)
        
        dense_cols = [
            "danceability", "energy", "speechiness", 
            "acousticness", "instrumentalness", "liveness", 
            "valence", "tempo"
        ]
        
        # 2. PRE-CONVERT to Numpy (Critical for Speed)
        # Polars row() access is slow. Numpy array access is instant.
        self.dense_features = df.select(dense_cols).to_numpy()
        self.genre_ids = df["genre_id"].to_numpy()
        self.mood_ids = df["mood_id"].to_numpy()
        
    def __len__(self):
        return len(self.genre_ids)
    
    def __getitem__(self, idx):
        # Access numpy arrays directly (O(1) speed)
        return {
            "dense_features": torch.tensor(self.dense_features[idx], dtype=torch.float32),
            "genre_id": torch.tensor(self.genre_ids[idx], dtype=torch.long),
            "mood_id": torch.tensor(self.mood_ids[idx], dtype=torch.long)
        }

def train():
    # 1. Load Data
    full_dataset = MusicDataset(DATA_PATH)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Num Workers=0 for Windows compatibility, =2 for Linux/Mac
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 2. Initialize Model
    num_genres = int(full_dataset.genre_ids.max() + 1)
    num_moods = int(full_dataset.mood_ids.max() + 1)
    feature_dim = full_dataset.dense_features.shape[1]
    
    model = TwoTowerModel(
        audio_feature_dim=feature_dim,
        num_genres=num_genres,
        num_moods=num_moods
    ).to(device) # Move model to GPU
    
    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MarginRankingLoss(margin=0.5) 

    print("Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # A. Move Batch to GPU/Device
            dense = batch['dense_features'].to(device)
            genres = batch['genre_id'].to(device)
            true_moods = batch['mood_id'].to(device)
            
            # B. Negative Sampling
            random_shift = torch.randint(low=1, high=num_moods, size=true_moods.shape).to(device)
            fake_moods = (true_moods + random_shift) % num_moods
            
            # C. Forward Pass
            pos_scores = model(dense, genres, true_moods)
            neg_scores = model(dense, genres, fake_moods)
            
            # D. Loss
            target = torch.ones(pos_scores.shape[0], 1).to(device)
            loss = criterion(pos_scores, neg_scores, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print occasional status update
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Step {batch_idx} | Loss: {loss.item():.4f}")
            
        print(f"--- Epoch {epoch+1} Completed | Avg Loss: {total_loss/len(train_loader):.4f} ---")
    
    # 4. Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()