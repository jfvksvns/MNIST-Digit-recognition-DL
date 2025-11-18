# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import io
import os
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load the trained model.
try:
    model = load_model("model/mnist_model_best.h5")
    print("Model loaded successfully.")
except:
    print("Warning: Could not load model. Make sure mnist_model.h5 is in the same directory.")
    model = None

def preprocess_image(img_bytes):
    """Enhanced preprocessing for MNIST model"""
    try:
        # Open image and convert to grayscale
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Adaptive preprocessing based on image characteristics
        img_mean = np.mean(img_array)
        
        # Check if we need to invert (MNIST has white digits on black background)
        if img_mean > 127:
            img_array = 255 - img_array
        
        # Apply thresholding to clean up the image
        # Use Otsu's method for automatic threshold
        _, img_array = cv2.threshold(
            img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Find bounding box of the digit to center it properly
        coords = cv2.findNonZero(img_array)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            
            # Add padding around the digit (20% of the size)
            padding = int(max(w, h) * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2 * padding)
            h = min(img_array.shape[0] - y, h + 2 * padding)
            
            # Crop to bounding box
            img_array = img_array[y:y+h, x:x+w]
        
        # Convert back to PIL for better resizing
        img = Image.fromarray(img_array)
        
        # Create a square canvas with padding (MNIST style)
        size = max(img.size)
        new_img = Image.new('L', (size, size), color=0)
        
        # Paste the digit in the center
        paste_x = (size - img.size[0]) // 2
        paste_y = (size - img.size[1]) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        # Resize to 28x28 using high-quality resampling
        new_img = new_img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(new_img).astype("float32") / 255.0
        
        # Apply slight Gaussian blur to smooth edges (helps with hand-drawn digits)
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        # Normalize again after blur
        if img_array.max() > 0:
            img_array = img_array / img_array.max()
        
        # Reshape for model (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    if not request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        file = next(iter(request.files.values()))
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)
        
        if img_array is None:
            return jsonify({"error": "Error processing image"}), 400
            
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_digit = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions, axis=1)[0])
        
        # Get top 3 predictions for better insight
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "digit": int(idx),
                "confidence": float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        return jsonify({
            "prediction": predicted_digit,
            "confidence": confidence,
            "top_3": top_3_predictions,
            "all_predictions": predictions[0].tolist()
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None
    })

@app.route("/", methods=["GET"])
def home():
    """Home endpoint with usage instructions"""
    return jsonify({
        "message": "MNIST Digit Recognition API (Enhanced)",
        "usage": "POST an image file to /predict endpoint",
        "tips": {
            "image_format": "PNG, JPG, or any common image format",
            "content": "Single digit, 0-9",
            "background": "Any color (will be auto-inverted if needed)",
            "quality": "Clear, centered digits work best"
        },
        "endpoints": {
            "GET /": "This info",
            "GET /health": "Health check",
            "POST /predict": "Predict digit from image"
        }
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)