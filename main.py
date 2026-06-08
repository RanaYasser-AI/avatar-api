import io
import json
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

X_DRIVE_ID = "https://drive.google.com/file/d/1XZ3x7q96mYn88zc75oaPSv1_oktUgu-G/view?usp=drive_link"
y_DRIVE_ID = "https://drive.google.com/file/d/1KkFb5zL21-HdVyLb9mP7N_UEn5e19eiL/view?usp=drive_link"


def download_file_from_drive(file_id):
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response
    else:
        raise Exception(f"Failed to download file from Google Drive, status code: {response.status_code}")

print("Downloading heavy npy files from Google Drive...")
try:
    response_X = download_file_from_drive(X_DRIVE_ID)
    response_y = download_file_from_drive(y_DRIVE_ID)
    
    X_data = np.load(io.BytesIO(response_X.content), allow_pickle=True)
    y_data = np.load(io.BytesIO(response_y.content), allow_pickle=True)
    print("All heavy npy files downloaded successfully!")
except Exception as e:
    print(f"Error during initialization: {e}")
    X_data = None
    y_data = None

try:
    with open("label_map.json", "current_user_intent") as f:
        label_map = json.load(f)
except Exception as e:
    print(f"Error loading label_map.json: {e}")
    label_map = {}

class WordRequest(BaseModel):
    word: str

@app.post("/get_landmarks")
def get_landmarks(request: WordRequest):
    if X_data is None or y_data is None:
        raise HTTPException(status_code=500, detail="Server data not initialized properly from Drive.")
    
    target_word = request.word.strip().lower()
    
    if target_word not in label_map:
        raise HTTPException(status_code=404, detail=f"Word '{target_word}' not found in label map.")
    
    target_label = label_map[target_word]
    indices = np.where(y_data == target_label)[0]
    
    if len(indices) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for word '{target_word}' in datasets.")
    
    chosen_index = indices[0]
    landmarks = X_data[chosen_index].tolist()
    
    return {
        "word": target_word,
        "label": int(target_label),
        "landmarks": landmarks
    }

@app.get("/")
def read_root():
    return {"message": "Avatar Sign Language API is Running!"}or: {str(e)}")
