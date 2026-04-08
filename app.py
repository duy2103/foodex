"""
Vietnamese Food Classification Model - Flask Web Application
Provides a modern UI to upload images and get predictions from the trained EfficientNetB2 model.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import json

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Class names from the Vietnamese Food Dataset
# IMPORTANT: Order MUST match ImageFolder alphabetical sorting
# This is the actual alphabetical order used by torchvision.datasets.ImageFolder
CLASS_NAMES = [
    'Bánh_cuốn',    # 0
    'Bánh_mì',      # 1
    'Bánh_xèo',     # 2
    'Bún_bò_Huế',   # 3
    'Bún_chả',      # 4
    'Bún_riêu',     # 5
    'Bánh_bao',     # 6
    'Cháo_lòng',    # 7
    'Chả_giò',      # 8
    'Chè',          # 9
    'Cơm_tấm',      # 10
    'Gỏi_cuốn',     # 11
    'Hủ_tiếu',      # 12
    'Mì_Quảng',     # 13
    'Phở'           # 14
]

# Display names for UI (user-friendly formatting)
DISPLAY_NAMES = {
    'Bánh_cuốn': 'Bánh cuốn',
    'Bánh_mì': 'Bánh mì',
    'Bánh_xèo': 'Bánh xèo',
    'Bún_bò_Huế': 'Bún bò Huế',
    'Bún_chả': 'Bún chả',
    'Bún_riêu': 'Bún riêu',
    'Bánh_bao': 'Bánh bao',
    'Cháo_lòng': 'Cháo lòng',
    'Chả_giò': 'Chả giò',
    'Chè': 'Chè',
    'Cơm_tấm': 'Cơm tấm',
    'Gỏi_cuốn': 'Gỏi cuốn',
    'Hủ_tiếu': 'Hủ tiếu',
    'Mì_Quảng': 'Mì Quảng',
    'Phở': 'Phở'
}

# Image preprocessing transforms (same as validation transforms)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Global state for model loading
model_loaded = False
model = None
model_error = None

# Load model
def load_model():
    """Load the trained EfficientNetB2 model"""
    global model_loaded, model, model_error
    
    try:
        # Create EfficientNetB2 model with matching architecture
        # Load without pre-trained weights first to avoid hash validation issues
        model_arch = models.efficientnet_b2(weights=None)
        
        # Replace classifier to match training setup
        num_ftrs = model_arch.classifier[1].in_features
        model_arch.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, len(CLASS_NAMES))
        )
        
        # Load saved weights
        model_path = os.path.join(os.path.dirname(__file__), 'vietnamese_food_efficientnetb2.pth')
        if os.path.exists(model_path):
            # Try loading with weights_only=False (bypasses hash validation issues)
            try:
                state_dict = torch.load(model_path, map_location=device, weights_only=False)
                model_arch.load_state_dict(state_dict)
                print(f"Model loaded successfully from {model_path}")
                model_error = None
            except Exception as e:
                print(f"Error loading model weights: {str(e)}")
                print(f"Warning: Model file could not be loaded. Using untrained model.")
                model_error = f"Model weights not found: {str(e)}"
        else:
            print(f"Warning: Model file not found at {model_path}. Using untrained model.")
            model_error = "Model weights not found. The model is using random initialization."
        
        model_arch = model_arch.to(device)
        model_arch.eval()
        model = model_arch
        model_loaded = True
        return model_arch
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model_error = f"Error loading model: {str(e)}"
        model_loaded = False
        return None

# Load model at startup
try:
    load_model()
except Exception as e:
    print(f"Critical error during model initialization: {str(e)}")
    model_error = str(e)
    model_loaded = False

# Confidence threshold for predictions (40% - images below this are considered "not in database")
CONFIDENCE_THRESHOLD = 85.0

def predict_image(image_path):
    """
    Predict the class of an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with predictions and confidence scores
    """
    try:
        # Check if model is loaded
        if model is None or not model_loaded:
            return {'error': 'Model is not loaded. Please wait for the model to load or check logs for errors.'}
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(outputs, 1)
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_percent = round(probabilities[predicted.item()].item() * 100, 2)
        
        # Check if confidence is below threshold
        if confidence_percent < CONFIDENCE_THRESHOLD:
            return {
                'not_in_database': True,
                'message': f'This image does not match any Vietnamese dish in our database.'
            }
        
        results = {
            'predicted_class': DISPLAY_NAMES[predicted_class],
            'confidence': confidence_percent,
            'top3_predictions': [
                {
                    'class': DISPLAY_NAMES[CLASS_NAMES[idx.item()]],
                    'confidence': round(prob.item() * 100, 2)
                }
                for prob, idx in zip(top3_probs, top3_indices)
            ]
        }
        return results
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/api/status', methods=['GET'])
def status():
    """Get the status of the model and app"""
    return jsonify({
        'model_loaded': model_loaded,
        'model_error': model_error,
        'device': str(device),
        'num_classes': len(CLASS_NAMES),
        'classes': CLASS_NAMES
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        # Check if image is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make prediction
        results = predict_image(file_path)
        
        # Convert image to base64 for display
        with open(file_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        results['image_base64'] = f"data:image/jpeg;base64,{image_data}"
        results['filename'] = filename
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of all food classes with display names"""
    display_classes = [DISPLAY_NAMES[cls] for cls in CLASS_NAMES]
    return jsonify({'classes': display_classes})

@app.route('/api/example/<food_class>', methods=['GET'])
def get_example(food_class):
    """Get an example image from the dataset for a given class"""
    try:
        # Map display name to folder name
        folder_name = food_class.replace(' ', '_')
        dataset_path = os.path.join(os.path.dirname(__file__), 'Vietnamese_Food_Dataset', folder_name)
        
        if os.path.exists(dataset_path):
            images = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                image_file = images[0]
                image_path = os.path.join(dataset_path, image_file)
                
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode()
                
                return jsonify({
                    'image_base64': f"data:image/jpeg;base64,{image_data}",
                    'filename': image_file
                })
        
        return jsonify({'error': f'No example found for {food_class}'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
