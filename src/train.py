import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import polars as pl
import numpy as np

# Import your custom modules
# (Ensure model.py is in the same folder or adjusted path)
from model import TwoTowerModel 
from torch.utils.data import Dataset

# --- Re-define Dataset to include Mood ID ---
class MusicDataset(Dataset):
    def __init__(self, parquet_file):
        self.data = pl.read_parquet(parquet_file)
        self.dense_cols = [
            "danceability", "energy", "speechiness", 
            "acousticness", "instrumentalness", "liveness", 
            "valence", "tempo"
        ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.row(idx, named=True)
        return {
            "dense_features": torch.tensor([row[col] for col in self.dense_cols], dtype=torch.float32),
            "genre_id": torch.tensor(row['genre_id'], dtype=torch.long),
            "mood_id": torch.tensor(row['mood_id'], dtype=torch.long) # Correct Mood (Target)
        }

# --- Training Configuration ---
DATA_PATH = "../data/processed/songs_final.parquet"
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64

def train():
    # 1. Load Data
    full_dataset = MusicDataset(DATA_PATH)
    
    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    # Calculate unique values dynamically
    num_genres = full_dataset.data["genre_id"].max() + 1
    num_moods = full_dataset.data["mood_id"].max() + 1
    
    model = TwoTowerModel(
        audio_feature_dim=len(full_dataset.dense_cols),
        num_genres=num_genres,
        num_moods=num_moods
    )
    
    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MarginRankingLoss(margin=0.5) 
    # Logic: Score(Positive) should be higher than Score(Negative) by at least 0.5 margin

    print("Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # A. Get Data
            dense = batch['dense_features']
            genres = batch['genre_id']
            true_moods = batch['mood_id']
            
            # B. Negative Sampling
            # Create a "fake" mood for every song to teach the model what is WRONG
            # We randomly shift the mood ID by a random amount to ensure it's different
            random_shift = torch.randint(low=1, high=num_moods, size=true_moods.shape)
            fake_moods = (true_moods + random_shift) % num_moods
            
            # C. Forward Pass (Positive)
            # How similar is the song to the CORRECT mood?
            pos_scores = model(dense, genres, true_moods)
            
            # D. Forward Pass (Negative)
            # How similar is the song to the WRONG mood?
            neg_scores = model(dense, genres, fake_moods)
            
            # E. Compute Loss
            # Target = 1 means "pos_scores" should be ranked higher than "neg_scores"
            target = torch.ones(pos_scores.shape[0], 1)
            loss = criterion(pos_scores, neg_scores, target)
            
            # F. Backprop
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")
    
    # 4. Save Model
    torch.save(model.state_dict(), "../src/two_tower_model.pth")
    print("Model saved to src/two_tower_model.pth")

if __name__ == "__main__":
    train()