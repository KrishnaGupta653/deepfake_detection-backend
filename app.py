from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import io
import os
import logging
from datetime import datetime
import uuid
from huggingface_hub import hf_hub_download

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])

# Configuration
class Config:
    HUGGINGFACE_REPO = os.getenv('HUGGINGFACE_REPO', 'krishnagupta365/deepfake_detector')
    MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'production_deepfake_detector.pth')
    IMG_SIZE = 224
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    UPLOAD_FOLDER = 'uploads'
    MODELS_FOLDER = 'models'

# Create directories
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.MODELS_FOLDER, exist_ok=True)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model class
class ProductionDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ProductionDeepfakeDetector, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_model():
    """Download model from Hugging Face"""
    try:
        logger.info(f"Downloading model from: {Config.HUGGINGFACE_REPO}")
        model_path = hf_hub_download(
            repo_id=Config.HUGGINGFACE_REPO,
            filename=Config.MODEL_FILENAME,
            cache_dir=Config.MODELS_FOLDER
        )
        logger.info(f"Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None

def load_model():
    """Load the model"""
    global model
    try:
        model = ProductionDeepfakeDetector()
        model_path = download_model()
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.to(device)
            model.eval()
            logger.info("✅ Model loaded successfully!")
            return True
        else:
            logger.error("❌ Model file not found")
            return False
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def allowed_file(filename):
    """Check file extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def preprocess_image(image_data):
    """Preprocess image"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Deepfake Detection API',
        'status': 'running',
        'model_loaded': model is not None
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check model
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read and check file size
        file_data = file.read()
        if len(file_data) > Config.MAX_FILE_SIZE:
            return jsonify({'error': 'File too large'}), 400
        
        # Preprocess image
        image_tensor = preprocess_image(file_data)
        if image_tensor is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            fake_prob = probabilities[0][0].item() * 100
            real_prob = probabilities[0][1].item() * 100
        
        # Determine prediction
        if real_prob > fake_prob:
            prediction = "Real"
            confidence = real_prob
        else:
            prediction = "Fake"
            confidence = fake_prob
        
        # Response
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'probabilities': {
                'fake': round(fake_prob, 2),
                'real': round(real_prob, 2)
            },
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'name': 'ResNet50',
                'version': '1.0',
                'accuracy': '98%+'
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500

# Initialize app
def create_app():
    """Application factory for gunicorn"""
    logger.info("🚀 Starting Deepfake Detection API")
    load_model()
    return app

if __name__ == '__main__':
    logger.info("🚀 Starting Deepfake Detection API")
    load_model()
    # Railway uses PORT environment variable
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"🌐 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
else:
    # For gunicorn - load model when imported
    load_model()
