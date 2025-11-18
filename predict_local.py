import pickle
import pandas as pd

# load trained model
with open('model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

# preprocess input
def preprocess_input(data):
    df = pd.DataFrame([data])

    cols_to_drop = [
        'Unnamed: 0',
        'track_id',
        'track_name',
        'album_name',
        'album_name',
        'track_name',
        'artists',
        'track_genre',
    ]
    
    # drop columns not used in model training
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # convert explicit column bool to numeric
    df["explicit"] = df["explicit"].astype(int)
    
    # Engineered numeric features
    df["hype_score"] = df["energy"] * df["danceability"]
    df["acoustic_softness"] = df["acousticness"] + df["instrumentalness"]
    df["duration_min"] = df["duration_ms"] / 60000
    df["beats_per_sec"] = df["tempo"] / 60
    df["is_high_energy"] = (df["energy"] > 0.7).astype(int)
    
    return df
    
# example song to test app
example = {
  "artists": "The Weeknd",
  "album_name": "After Hours",
  "track_name": "Blinding Lights",
  "track_id": "0VjIjW4GlUZAMYd2vXMi3b",
  "duration_ms": 200040,
  "explicit": False,
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

# make predictions
track = preprocess_input(example)
prediction = model.predict(track)[0]
result = round(float(prediction), 2)
print(f'The expected popularity of the song "{example['track_name']}" by "{example['artists']}" is: {result}')