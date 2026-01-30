import streamlit as st
import torch
import polars as pl
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Import your model definition
# (Make sure model.py is in the same folder or Python path)
from model import TwoTowerModel

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/processed/songs_final.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "two_tower_model.pth")
ENCODER_PATH = os.path.join(BASE_DIR, "../data/processed/mood_encoder.pkl")
DEVICE = torch.device("cpu") # CPU is fine for inference

# --- LOAD ARTIFACTS (Cached for Speed) ---
@st.cache_resource
def load_system():
    # 1. Load Data
    df = pl.read_parquet(DATA_PATH)
    
    # 2. Load Encoder
    with open(ENCODER_PATH, "rb") as f:
        mood_encoder = pickle.load(f)
        
    # 3. Load Model
    num_genres = df["genre_id"].max() + 1
    num_moods = len(mood_encoder.classes_)
    
    model = TwoTowerModel(
        audio_feature_dim=8, 
        num_genres=num_genres,
        num_moods=num_moods
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 4. Pre-compute Item Embeddings (The "Vector DB")
    print("Indexing Songs...")
    dense_cols = [
        "danceability", "energy", "speechiness", 
        "acousticness", "instrumentalness", "liveness", 
        "valence", "tempo"
    ]
    dense_feats = torch.tensor(df.select(dense_cols).to_numpy(), dtype=torch.float32)
    genre_ids = torch.tensor(df["genre_id"].to_numpy(), dtype=torch.long)
    
    with torch.no_grad():
        song_vectors = model.item_tower(dense_feats, genre_ids).numpy()
        
    return df, model, mood_encoder, song_vectors

# --- INIT APP ---
st.set_page_config(page_title="MoodTunes AI", page_icon="🎵")
st.title("🎵 MoodTunes AI: Context-Aware Recommender")
st.markdown("I used a **Two-Tower Neural Network** to align Audio Features with User Moods.")

# Load everything
try:
    df, model, mood_encoder, song_vectors = load_system()
    st.success("System Loaded! Model & Vector Index Ready.")
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# --- SIDEBAR: USER CONTEXT ---
st.sidebar.header("Your Context")
selected_mood = st.sidebar.selectbox("How are you feeling?", mood_encoder.classes_)

# Optional: Show "Under the Hood" details
show_details = st.sidebar.checkbox("Show Technical Details")

# --- MAIN INFERENCE ENGINE ---
if st.button("Generate Recommendations"):
    
    # 1. Encode User Query
    mood_idx = mood_encoder.transform([selected_mood])[0]
    mood_tensor = torch.tensor([mood_idx], dtype=torch.long)
    
    # 2. Get Query Vector from Query Tower
    with torch.no_grad():
        query_vec = model.query_tower(mood_tensor).numpy()
    
    # 3. Vector Search (Dot Product)
    # Calculate similarity between Query Vector and ALL 100k+ Song Vectors
    scores = cosine_similarity(query_vec, song_vectors)
    
    # 4. Get Top 10
    top_k_indices = np.argsort(scores[0])[::-1][:10]
    results = df[top_k_indices]
    
    # --- DISPLAY RESULTS ---
    st.subheader(f"Top 10 Picks for '{selected_mood}' Vibes")
    
    for row in results.iter_rows(named=True):
        # Create a card-like display
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"**{row['mood_label']}**")
            with col2:
                st.markdown(f"**{row['track_name']}** by *{row['artist_name']}*")
                if show_details:
                    st.caption(f"Genre: {row['genre']} | Energy: {row['energy']:.2f} | Valence: {row['valence']:.2f}")
            st.divider()

    # --- TECHNICAL VIZ (For Recruiters) ---
    if show_details:
        st.subheader("Under the Hood: Vector Space")
        st.write("Query Vector (First 8 dims):")
        st.code(str(query_vec[0][:8]), language="python")