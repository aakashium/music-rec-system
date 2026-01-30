# MoodTunes AI: Context-Aware Music Recommender

> **A Two-Tower Neural Network that recommends music based on user mood, optimizing for a 95% Hit Rate.**

## Project Overview
Traditional recommender systems rely on Collaborative Filtering (User A likes User B), which fails with new users (Cold Start Problem). **MoodTunes AI** solves this by using a **Content-Based Two-Tower Architecture**. It maps **Audio Features** (Energy, Valence, Tempo) and **User Context** (Current Mood) into a shared Vector Space using Contrastive Learning.

## System Architecture
* **Data Pipeline:** Polars for high-performance ETL on the Spotify 1M Dataset.
* **Model:** PyTorch **Two-Tower Network** (Query Tower + Item Tower).
* **Training:** Optimized using **Margin Ranking Loss** with Negative Sampling.
* **Inference:** Vector Search (Cosine Similarity) for <50ms retrieval latency.
* **Deployment:** Streamlit UI with cached embeddings for real-time interaction.

## Performance (Metric: Hit Rate @ 10)
I evaluated the model by querying specific moods and checking the relevance of the Top 10 results.
* **Happy:** 100% Accuracy
* **Sad:** 100% Accuracy
* **Angry:** 100% Accuracy
* **Calm:** 60% Accuracy (Identified area for improvement: overlap with 'Sad' vectors)

## Tech Stack
* **ML Core:** PyTorch, Scikit-Learn
* **Data Engineering:** Polars, Numpy
* **App/Serving:** Streamlit
* **Environment:** UV (Rust-based Python package manager)

## How to Run Locally
1. Clone the repo
2. Install dependencies:
```bash
uv add -r requirements.txt
```
3. Run the App:
```bash
streamlit run src/app.py
```
