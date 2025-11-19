# 🎵 Spotify Song Popularity Prediction

## Overview

This project predicts the **popularity of songs on Spotify** based on their audio features and metadata. The goal is to provide a **predictive model and a simple API** that can estimate a song’s popularity score before it is released, allowing artists, producers, and music platforms to make informed decisions.

---

## Problem Statement

Spotify provides various audio features for each track, such as danceability, energy, tempo, and more. Understanding which features contribute most to a song's popularity can help:

- Artists and producers optimize tracks for audience appeal.  
- Music streaming platforms recommend songs more effectively.  
- Music marketers plan campaigns around tracks that have high predicted popularity.  

The problem is formulated as a **regression task**: given the characteristics of a track, predict its **popularity score** (0–100) as measured by Spotify.

---

## Dataset

The dataset used is publicly available on [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). It contains:

- **Audio features:** danceability, energy, tempo, loudness, acousticness, instrumentalness, valence, etc.  
- **Metadata:** track name, album, artists, genre, duration, explicit flag, key, mode, time signature.  
- **Target:** popularity score (0–100).  

**Note:** The dataset has roughly 1,000 songs per genre, spanning 114 genres.

---

## Solution Approach

1. **Data Preprocessing:**
   - Drop irrelevant identifiers (track_id, track_name, album_name, artists).  
   - Convert boolean columns (`explicit`) to numeric.  
   - Engineer additional features, e.g., hype_score, acoustic_softness, duration in minutes, beats per second.  

2. **Feature Encoding:**
   - Target encoding for features like `track_genre` to handle high-cardinality categorical variables.  

3. **Modeling:**
   - Several regression models were tested: Random Forest, XGBoost, LightGBM.
   - XGBoost model was chosen based on speed and performance.  
   - Hyperparameters optimized using GridSearchCV and cross-validation.  
   - Best-performing XGBoostRegressor model selected based on RMSE on a validation set.  

4. **Deployment:**
   - The trained model is served through a **FastAPI** application.  
   - API accepts JSON input representing a track and returns predicted popularity.  
   - Containerized with **Docker** for easy deployment and scalability.  

---

## API Usage

**Endpoint:** `POST /predict`  

**Request Body Example:**

```json
{
  "artists": "The Weeknd",
  "album_name": "After Hours",
  "track_name": "Blinding Lights",
  "track_id": "0VjIjW4GlUZAMYd2vXMi3b",
  "duration_ms": 200040,
  "explicit": false,
  "danceability": 0.51,
  "energy": 0.73,
  "key": 1,
  "loudness": -5.934,
  "mode": 1,
  "speechiness": 0.059,
  "acousticness": 0.0015,
  "instrumentalness": 0.0,
  "liveness": 0.0896,
  "valence": 0.334,
  "tempo": 171.005,
  "time_signature": 4,
  "track_genre": "pop"
}
```
---

## (Optional) Running Locally

### 1. Set Up a Python Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Activate the environment (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies with uv

```bash
# Install uv if not already installed
pip install uv

# Install dependencies from the lock file
uv sync --locked
```

### 3. (Optional) Create a jupyter kernel to run the notebooks
```bash
python -m ipykernel install --user --name="name_of_venv" --display-name "display_name_of_venv"
```

---

## Using Docker

The application is already containerized. You can use the Docker image provided in this repository to run the FastAPI app without installing dependencies locally.

### 1. Build the Docker Image

If the Dockerfile is included in your repo, build the image with:

```bash
docker build -t song-popularity .
```

