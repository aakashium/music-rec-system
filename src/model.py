import torch
import torch.nn as nn
import torch.nn.functional as F

class ItemTower(nn.Module):
    """
    Encodes song features into a vector embedding.
    Input: Numerical Audio Features (Standardized) + Genre ID
    """
    def __init__(self, input_dim, num_genres, embedding_dim=32, output_dim=64):
        super().__init__()
        
        # 1. Genre Embedding (Categorical -> Dense Vector)
        self.genre_embedding = nn.Embedding(num_embeddings=num_genres, embedding_dim=embedding_dim)
        
        # 2. Dense Layers (Process Audio Features + Genre)
        # Input size = (Audio Features) + (Genre Embedding)
        self.fc1 = nn.Linear(input_dim + embedding_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc_out = nn.Linear(64, output_dim) # Final Embedding Size

    def forward(self, dense_features, genre_ids):
        # Look up embedding for the genre
        genre_vec = self.genre_embedding(genre_ids) 
        
        # Concatenate Audio Features with Genre Vector
        x = torch.cat([dense_features, genre_vec], dim=1)
        
        # Pass through layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        
        # Normalize the final vector (Critical for Cosine Similarity!)
        x = self.fc_out(x)
        return F.normalize(x, p=2, dim=1)

class QueryTower(nn.Module):
    """
    Encodes the User's Intent (Context) into a vector.
    Input: The Target Mood (e.g., Happy, Sad) encoded as integers.
    """
    def __init__(self, num_moods, embedding_dim=16, output_dim=64):
        super().__init__()
        
        # Embedding for Mood (Happy=0, Sad=1, etc.)
        self.mood_embedding = nn.Embedding(num_embeddings=num_moods, embedding_dim=embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc_out = nn.Linear(64, output_dim)

    def forward(self, mood_ids):
        x = self.mood_embedding(mood_ids)
        x = F.relu(self.fc1(x))
        x = self.fc_out(x)
        return F.normalize(x, p=2, dim=1)

class TwoTowerModel(nn.Module):
    """
    Combines both towers. 
    During training, we try to maximize similarity between Query and relevant Item.
    """
    def __init__(self, audio_feature_dim, num_genres, num_moods):
        super().__init__()
        self.item_tower = ItemTower(audio_feature_dim, num_genres)
        self.query_tower = QueryTower(num_moods)
        
    def forward(self, audio_features, genre_ids, mood_ids):
        # Generate embeddings from both towers
        item_embedding = self.item_tower(audio_features, genre_ids)
        query_embedding = self.query_tower(mood_ids)
        
        # Calculate Similarity (Dot Product)
        # Result is a score: How well does this song fit this mood?
        similarity = torch.sum(item_embedding * query_embedding, dim=1, keepdim=True)
        return similarity
        