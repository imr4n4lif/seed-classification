import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Clean TensorFlow logs

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ✅ CORRECT preprocessing for EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input


# =========================
# App initialization
# =========================
app = FastAPI(title="Seed Classification API - EfficientNet")


# =========================
# CORS (for web frontend)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for local dev
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Load trained model
# =========================
# EfficientNet expects specific input sizes depending on the variant:
# EfficientNetB0: 224x224
# EfficientNetB1: 240x240
# EfficientNetB2: 260x260
# EfficientNetB3: 300x300
# EfficientNetB4: 380x380
# EfficientNetB5: 456x456
# EfficientNetB6: 528x528
# EfficientNetB7: 600x600

# Update this to match your model variant
IMG_SIZE = (224, 224)  # For EfficientNetB0

try:
    model = tf.keras.models.load_model(
        "effnetb0_finetune.h5",  # Update with your model filename
        custom_objects={"preprocess_input": preprocess_input}
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure 'efficientnet_finetune.h5' exists in the current directory")
    model = None


# =========================
# Class labels (14 classes)
# =========================
class_names = [
    "Bitter melon",
    "Bottle gourd",
    "Carrot",
    "Cauliflower",
    "Chili",
    "Coriander leaves",
    "Cucumber",
    "Hyacinth bean",
    "Malabar spinach",
    "Onion",
    "Radish",
    "Spinach",
    "Tomato",
    "Water spinach",
]


# =========================
# Health / info endpoint
# =========================
@app.get("/")
async def root():
    return {
        "message": "Seed Classification API (EfficientNet)",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
        "classes": class_names,
        "input_size": IMG_SIZE,
    }


# =========================
# Model info endpoint
# =========================
@app.get("/model-info")
async def model_info():
    if model is None:
        return JSONResponse(
            {"error": "Model not loaded"},
            status_code=503,
        )
    
    try:
        return {
            "architecture": "EfficientNet",
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "total_params": model.count_params(),
            "num_classes": len(class_names),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if model is loaded
    if model is None:
        return JSONResponse(
            {"error": "Model not loaded. Check server logs."},
            status_code=503,
        )
    
    try:
        # 🔒 Validate file type
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                {"error": "Invalid file type. Please upload an image."},
                status_code=400,
            )

        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to match model input size
        img = img.resize(IMG_SIZE)

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ REQUIRED for EfficientNet
        # EfficientNet preprocessing normalizes to [-1, 1] range
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][idx])

        # Get top 5 predictions
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [
            {
                "class": class_names[i],
                "confidence": float(predictions[0][i])
            }
            for i in top_5_idx
        ]

        return {
            "predicted_class": class_names[idx],
            "confidence": confidence,
            "all_probabilities": predictions[0].tolist(),
            "top_5": top_5_predictions,
            "model": "EfficientNet",
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# Batch prediction endpoint (optional)
# =========================
@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if model is None:
        return JSONResponse(
            {"error": "Model not loaded. Check server logs."},
            status_code=503,
        )
    
    if len(files) > 10:
        return JSONResponse(
            {"error": "Maximum 10 images allowed per batch"},
            status_code=400,
        )
    
    try:
        results = []
        
        for file in files:
            if not file.content_type.startswith("image/"):
                results.append({
                    "filename": file.filename,
                    "error": "Invalid file type"
                })
                continue
            
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img = img.resize(IMG_SIZE)
            
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            predictions = model.predict(img_array, verbose=0)
            idx = int(np.argmax(predictions[0]))
            
            results.append({
                "filename": file.filename,
                "predicted_class": class_names[idx],
                "confidence": float(predictions[0][idx]),
            })
        
        return {"results": results, "total": len(results)}
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# Run server
# =========================
if __name__ == "__main__":
    print(f"🚀 Starting EfficientNet Seed Classification API")
    print(f"📊 Image size: {IMG_SIZE}")
    print(f"🏷️  Classes: {len(class_names)}")
    uvicorn.run(app, host="0.0.0.0", port=8000)