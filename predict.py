import pickle
import pandas as pd

from typing import Literal
from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI

# Setup APP
app = FastAPI(title="song-popularity-prediction")

class Track(BaseModel):
    artists: str = Field(..., example="The Weeknd")
    album_name: str = Field(..., example="After Hours")
    track_name: str = Field(..., example="Blinding Lights")
    track_id: str = Field(..., example="0VjIjW4GlUZAMYd2vXMi3b")

    duration_ms: int = Field(..., ge=0, example=200040)
    explicit: bool = Field(..., example=False)

    danceability: float = Field(..., ge=0, le=1, example=0.51)
    energy: float = Field(..., ge=0, le=1, example=0.73)
    key: int = Field(..., ge=0, le=11, example=1)
    loudness: float = Field(..., example=-5.934)
    mode: Literal[0, 1]
    speechiness: float = Field(..., ge=0, le=1, example=0.059)
    acousticness: float = Field(..., ge=0, le=1, example=0.0015)
    instrumentalness: float = Field(..., ge=0, le=1, example=0.0)
    liveness: float = Field(..., ge=0, le=1, example=0.0896)
    valence: float = Field(..., ge=0, le=1, example=0.334)
    tempo: float = Field(..., ge=0, example=171.005)
    time_signature: Literal[1, 2, 3, 4, 5] = Field(..., example=4)

    track_genre: str = Field(..., example="pop")


class PredictPopularity(BaseModel):
    song_popularity: float


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
        'artists',
        'album_name',
        'track_name',
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


def predict_popularity(example):
    track = preprocess_input(example)
    prediction = model.predict(track)[0]
    result = round(float(prediction), 2)
    return result


@app.post("/predict")
def predict(example: Track) -> PredictPopularity:
    data = example.model_dump()
    result = predict_popularity(data)

    print(f"The expected popularity of the song '{example.track_name}' by '{example.artists}' is: {result}")
        
    return PredictPopularity(
        song_popularity=result
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)