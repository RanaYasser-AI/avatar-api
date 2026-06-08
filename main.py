import os
import json
import numpy as np
import requests
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

X_DRIVE_ID = "1XZ3x7q96mYn88zc75oaPSv1_oktUgu-G"
y_DRIVE_ID = "1KkFb5zL21-HdVyLb9mP7N_UEn5e19eiL"

# 🔄 الدالة الجديدة والمضمونة لتنزيل الملفات الكبيرة وتخطي حماية جوجل درايف
def download_file_from_drive(file_id, destination):
    if os.path.exists(destination) and os.path.getsize(destination) > 1000:
        return
        
    print(f"Downloading {destination} from Google Drive...")
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # طلب أول عشان نشوف هل فيه توكن تأكيد (Confirmation Token) للملفات الكبيرة
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    # لو لقرينا التوكن، بنبعت طلب تاني بيه عشان نتخطى صفحة التحذير
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        
    # حفظ الملف الحقيقي بالكامل
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

@app.on_event("startup")
def startup_event():
    try:
        download_file_from_drive(X_DRIVE_ID, os.path.join(AVATAR_DATA_DIR, "X.npy"))
        download_file_from_drive(y_DRIVE_ID, os.path.join(AVATAR_DATA_DIR, "y.npy"))
        print("All heavy npy files downloaded successfully!")
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
            
        X_data = np.load(os.path.join(AVATAR_DATA_DIR, "X.npy"), allow_pickle=True)
        y_data = np.load(os.path.join(AVATAR_DATA_DIR, "y.npy"), allow_pickle=True)
        
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
