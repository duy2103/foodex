# 🍜 Vietnamese Food Classifier

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

A modern, beautiful web interface for Vietnamese Food Classification powered by ResNet50 deep learning. Upload an image of Vietnamese cuisine, and the AI model will instantly identify the dish with high accuracy.

## ✨ Features

### 🎨 Modern & Responsive UI
- Clean, gradient-based design with smooth animations
- Drag-and-drop image upload with preview
- Real-time predictions with confidence scores
- Mobile-friendly responsive layout
- Beautiful category gallery with hover effects

### 🤖 AI-Powered Predictions
- ResNet50 transfer learning model (pre-trained on ImageNet)
- 15 Vietnamese food categories
- Top-3 prediction confidence display
- Instant classification with visual feedback
- GPU acceleration (CUDA/MPS) with automatic fallback to CPU

### 📸 Image Gallery
- Browse sample images from each food category
- Click any category to load example dishes
- High-quality image preprocessing
- Beautiful grid layout with lazy loading

## 📁 Project Structure

```
minicp/
├── app.py                              # Flask backend (main application)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── templates/
│   ├── index.html                      # Main web UI interface
│   └── modern_ui.html                  # Alternative modern design
├── uploads/                            # User-uploaded images (auto-created)
├── Vietnamese_Food_Dataset/            # Training dataset (~500-1000 images per category)
│   ├── Bánh_bao/
│   ├── Bánh_cuốn/
│   ├── Bánh_mì/
│   ├── ... (15 food categories)
│   └── Phở/
├── vietnamese_food_resnet50.pth        # Trained model weights (ResNet50)
├── vietnamese_food_efficientnetb2.pth  # Alternative model weights (EfficientNetB2)
├── model.ipynb                         # Model training & evaluation notebook
├── datacollection.ipynb                # Data collection & preprocessing notebook
└── __pycache__/                        # Python cache (auto-generated)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager
- (Optional) CUDA 11.8+ for GPU support

### Installation

**1. Clone the Repository**
```bash
git clone https://github.com/duy2103/foodex.git
cd foodex
```

**2. Create Virtual Environment**
```bash
# Using venv (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n vietnamese-food python=3.10
conda activate vietnamese-food
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**PyTorch Installation Notes:**
```bash
# CPU only (recommended for beginners)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Mac (Apple Silicon/M1/M2/M3)
pip install torch torchvision
```

**4. Download Model Weights**

Ensure the trained model file exists in the project root:
- `vietnamese_food_resnet50.pth` (Primary model - ResNet50)
- `vietnamese_food_efficientnetb2.pth` (Alternative - EfficientNetB2)

If missing, download from the [Releases](https://github.com/duy2103/foodex/releases) page.

## 🎯 Usage

### Start the Application

```bash
python app.py
```

Expected output:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

### How to Use

1. **📤 Upload an Image**
   - Click the upload area or drag-and-drop a food image
   - Supported formats: JPG, PNG, WebP
   - Maximum file size: 16MB

2. **🔍 Analyze**
   - Click "Analyze Image" button
   - Wait for the AI model to process (usually < 2 seconds)
   - View the predicted food class and confidence score

3. **📊 View Results**
   - See the main prediction with confidence percentage
   - Check top 3 predictions with their confidence scores
   - Visual confidence bars for easy comparison

4. **📚 Explore Examples**
   - Browse the food categories gallery
   - Click any category to load sample images from the dataset
   - Learn from training examples

## 🍽️ Supported Food Categories

The model classifies 15 authentic Vietnamese dishes:

| # | Dish | Vietnamese Name | Emoji |
|---|------|-----------------|-------|
| 1 | Steamed Buns | Bánh bao | 🥟 |
| 2 | Rolled Pancakes | Bánh cuốn | 🥙 |
| 3 | Vietnamese Sandwich | Bánh mì | 🥪 |
| 4 | Sizzling Pancakes | Bánh xèo | 🥘 |
| 5 | Huế Beef Noodles | Bún bò Huế | 🍲 |
| 6 | Grilled Pork Noodles | Bún chả | 🍜 |
| 7 | Crab Noodle Soup | Bún riêu | 🍲 |
| 8 | Spring Rolls | Chả giò | 🥟 |
| 9 | Organ Congee | Cháo lòng | 🍲 |
| 10 | Vietnamese Dessert Drinks | Chè | 🧋 |
| 11 | Broken Rice | Cơm tấm | 🍚 |
| 12 | Fresh Spring Rolls | Gỏi cuốn | 🥗 |
| 13 | Clear Noodle Soup | Hủ tiếu | 🍲 |
| 14 | Quang Noodles | Mì Quảng | 🍝 |
| 15 | Beef Noodle Soup | Phở | 🍲 |

## 🔌 API Endpoints

### 📤 Upload & Predict
```http
POST /api/predict
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
```

**Response:**
```json
{
  "predicted_class": "Phở",
  "confidence": 95.23,
  "image_base64": "data:image/jpeg;base64,...",
  "filename": "image.jpg",
  "top3_predictions": [
    {"class": "Phở", "confidence": 95.23},
    {"class": "Bún chả", "confidence": 3.45},
    {"class": "Bánh mì", "confidence": 1.32}
  ]
}
```

### 📋 Get All Classes
```http
GET /api/classes
```

**Response:**
```json
{
  "classes": ["Bánh bao", "Bánh cuốn", "Bánh mì", ...]
}
```

### 🖼️ Get Example Image
```http
GET /api/example/<food_class>
```

**Response:**
```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "filename": "000001.jpg"
}
```

## ⚡ Performance Tips

- **First Request**: May be slower (model loading from disk) - ~3-5 seconds
- **Subsequent Requests**: Fast predictions (model cached in memory) - <2 seconds
- **GPU Support**: Automatically detects CUDA/MPS for faster inference (10-50x speedup)
- **CPU Mode**: Works on all machines, just slower than GPU
- **Image Size**: Automatically resized to 224x224 (optimal for ResNet50)

## 🐛 Troubleshooting

### Model File Not Found
```
Error: Model file not found at vietnamese_food_resnet50.pth
```
**Solution**: 
- Ensure the model file is in the project root directory
- Download from [Releases](https://github.com/duy2103/foodex/releases) if missing
- Verify you're running from the correct folder: `pwd` should show `.../minicp`

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Change the port in `app.py`:
```python
# Around line 200 in app.py
app.run(debug=True, host='0.0.0.0', port=5001)  # Use 5001 instead of 5000
```

Or kill the process:
```bash
# Find process on port 5000
lsof -i :5000

# Kill it
kill -9 <PID>
```

### CUDA/GPU Not Detected
```
Info: Using CPU for inference
```
**Solution**: This is normal - the app automatically falls back to CPU if GPU is unavailable. Just slower, not an error. For CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Image Upload Fails
**Checklist:**
- ✅ File size ≤ 16MB
- ✅ Format: JPG, PNG, or WebP
- ✅ Check browser console for detailed error (F12 → Console)
- ✅ Try different image if corrupted
- ✅ Clear browser cache and reload

### Import Errors
```
ModuleNotFoundError: No module named 'flask'
```
**Solution**: Activate virtual environment and reinstall dependencies:
```bash
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## 📚 Model Information

| Property | Value |
|----------|-------|
| **Architecture** | ResNet50 (transfer learning) |
| **Pre-training** | ImageNet-1K |
| **Input Size** | 224×224 pixels |
| **Classes** | 15 Vietnamese food categories |
| **Framework** | PyTorch 2.1.0 |
| **Training Data** | ~500-1000 images per category |
| **Model Size** | ~100 MB |

### Training Details
The model was trained using:
- **Transfer Learning**: Pre-trained ResNet50 from ImageNet fine-tuned on Vietnamese food
- **Data Augmentation**: Color jitter, rotation, random crop, horizontal flip
- **Optimization**: Adam optimizer with learning rate scheduling
- **Validation**: Stratified train/validation split (80/20)
- **Early Stopping**: Prevents overfitting after convergence
- **Accuracy**: ~92% top-1 accuracy on validation set

See `model.ipynb` for detailed training code and performance metrics.

## 🔧 Development

### Project Setup for Developers

```bash
git clone https://github.com/duy2103/foodex.git
cd foodex
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run in Development Mode
```bash
export FLASK_ENV=development
python app.py
```

### Code Structure
- `app.py`: Flask application with prediction API and web routes
- `templates/index.html`: Main UI with drag-drop upload and gallery
- `model.ipynb`: Model training pipeline with data augmentation
- `datacollection.ipynb`: Data collection and preprocessing workflow

### Making Changes
1. **UI Changes**: Edit `templates/index.html`
2. **API Changes**: Modify functions in `app.py`
3. **Model Changes**: Update code in `model.ipynb` and retrain

## 🚢 Production Deployment

### Using Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker
```bash
docker build -t vietnamese-food-classifier .
docker run -p 5000:5000 vietnamese-food-classifier
```

### Cloud Deployment Options
- **Heroku**: `git push heroku main`
- **AWS EC2**: Deploy with Docker or Gunicorn
- **Google Cloud Run**: Containerized deployment
- **Azure App Service**: Web app deployment

## 📝 Future Enhancements

- [ ] Batch image upload & prediction
- [ ] Download prediction results as CSV/PDF
- [ ] Confidence threshold adjustment UI
- [ ] Real-time webcam predictions
- [ ] Recipe suggestions based on predictions
- [ ] Multi-language support (English, Vietnamese, etc.)
- [ ] Docker containerization
- [ ] Mobile app (React Native/Flutter)
- [ ] Model quantization for faster inference
- [ ] User feedback & model retraining pipeline

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

Please ensure:
- Code follows PEP 8 style guide
- Tests pass and new tests are added
- README is updated if needed
- Commit messages are descriptive

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Free to use for educational, commercial, and personal projects.

## 👤 Author

**Duy Vu** - [@duy2103](https://github.com/duy2103)

- 🎓 Education: AI/ML enthusiast
- 🌍 Location: Vietnam
- 💼 Portfolio: [GitHub Profile](https://github.com/duy2103)

## 🙏 Acknowledgments

- **ResNet50** architecture from [torchvision](https://pytorch.org/vision/)
- **PyTorch** deep learning framework
- Vietnamese food dataset from web sources
- Inspired by modern food classification projects
- Thanks to the open-source community

## 📞 Support & Contact

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/duy2103/foodex/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/duy2103/foodex/discussions)

---

<div align="center">

**Made with ❤️ for Vietnamese Food Lovers** 🍜

[⭐ Star This Project](https://github.com/duy2103/foodex) | [🐛 Report Bug](https://github.com/duy2103/foodex/issues) | [✨ Request Feature](https://github.com/duy2103/foodex/issues)

</div>
