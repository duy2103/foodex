"""
Fix model file by removing hash validation issues
This script loads the model with weights_only=False and resaves it
"""

import torch
import torch.nn as nn
from torchvision import models
import os

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Class names
CLASS_NAMES = [
    'Bánh_cuốn', 'Bánh_mì', 'Bánh_xèo', 'Bún_bò_Huế', 'Bún_chả', 'Bún_riêu',
    'Bánh_bao', 'Cháo_lòng', 'Chả_giò', 'Chè', 'Cơm_tấm', 'Gỏi_cuốn',
    'Hủ_tiếu', 'Mì_Quảng', 'Phở'
]

print("Fixing model file...")

# Create model architecture
model_arch = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
num_ftrs = model_arch.classifier[1].in_features
model_arch.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, len(CLASS_NAMES))
)

model_path = os.path.join(os.path.dirname(__file__), 'vietnamese_food_efficientnetb2.pth')

try:
    # Load the corrupted model with weights_only=False
    print(f"Loading model from {model_path} (with weights_only=False)...")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model_arch.load_state_dict(state_dict)
    print("✓ Model loaded successfully")
    
    # Resave the model in a clean state
    temp_path = model_path + '.fixed'
    print(f"Saving fixed model to {temp_path}...")
    torch.save(model_arch.state_dict(), temp_path)
    print("✓ Model saved")
    
    # Verify the fixed model can be loaded
    print("Verifying fixed model...")
    model_test = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    model_test.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, len(CLASS_NAMES))
    )
    model_test.load_state_dict(torch.load(temp_path, map_location=device, weights_only=True))
    print("✓ Fixed model verified successfully!")
    
    # Replace original with fixed
    os.replace(temp_path, model_path)
    print(f"✓ Original model file replaced with fixed version")
    print("\n✅ Model file fixed successfully!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("\nFallback: The app will use weights_only=False to load the model")
