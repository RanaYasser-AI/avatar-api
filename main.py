import os
import json
import numpy as np
import requests
import gdown
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Avatar Animation Bridge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AVATAR_DATA_DIR = "."

# حطيتلك هنا الـ ID بتاع الملف المضغوط الجديد اللي لسه رافعينه على الدرايف سوا
X_ZIP_DRIVE_ID = "1vQqzlatAaPM3B0hDfGiNZwjXUQzXZo8n" 

def download_file_from_drive(file_id, destination):
    if os.path.exists(destination):
        return
    print(f"Downloading {destination} from Google Drive...")
    # تعديل بسيط عشان gdown يشتغل على السيرفرات بكفاءة أكبر
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)
    print(f"Downloaded: {destination}")

@app.on_event("startup")
def startup_event():
    try:
        # بننزل الملف المضغوط الجديد بس من الدرايف
        download_file_from_drive(X_ZIP_DRIVE_ID, os.path.join(AVATAR_DATA_DIR, "x_compressed.npz"))
        print("Compressed X file downloaded successfully!")
    except Exception as e:
        print(f"Error downloading files: {str(e)}")

@app.get("/")
def home():
    return {"status": "Avatar Animation Server is running successfully!"}

@app.get("/avatar/{word}")
def get_avatar_landmarks(word: str):
    try:
        label_map_path = os.path.join(AVATAR_DATA_DIR, "label_map.json")
        with open(label_map_path, "r", encoding="utf-8") as f:
            avatar_labels = json.load(f)
            
        target_word = word.strip().lower()
        avatar_index = None
        for key, value in avatar_labels.items():
            if str(value).strip().lower() == target_word:
                avatar_index = int(key)
                break
                
        if avatar_index is None:
            raise HTTPException(status_code=404, detail=f"Word '{word}' not found in Avatar dataset.")
            
        # --- التعديل السحري هنا ---
        # 1. بنقرا الـ X من الملف المضغوط أوتوماتيك بدون ما ياخد مساحة
        loaded_x = np.load(os.path.join(AVATAR_DATA_DIR, "x_compressed.npz"))
        X_data = loaded_x['data']
        
        # 2. بنقرا الـ y اللي هو كدة كدة مرفوع وموجود مع المشروع على جيت هاب
        y_data = np.load(os.path.join(AVATAR_DATA_DIR, "y.npy"))
        
        indices = np.where(y_data == avatar_index)[0]
        if len(indices) == 0:
            raise HTTPException(status_code=404, detail="No animation frames found for this word.")
            
        animation_sequence = X_data[indices[0]].tolist()
        
        return {
            "word": avatar_labels[str(avatar_index)],
            "avatar_label_index": avatar_index,
            "frames": X_data.shape[1],
            "landmarks_per_frame": X_data.shape[2],
            "landmarks": animation_sequence
        }
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")