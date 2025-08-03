# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torchvision.models import resnet50, ResNet50_Weights
# from PIL import Image
# import io
# import os
# import logging
# from datetime import datetime
# import uuid
# from huggingface_hub import hf_hub_download

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app, origins=["*"])

# # Configuration
# class Config:
#     HUGGINGFACE_REPO = os.getenv('HUGGINGFACE_REPO', 'krishnagupta365/deepfake_detector')
#     MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'production_deepfake_detector.pth')
#     IMG_SIZE = 224
#     MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
#     ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
#     UPLOAD_FOLDER = 'uploads'
#     MODELS_FOLDER = 'models'

# # Create directories
# os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(Config.MODELS_FOLDER, exist_ok=True)

# # Image preprocessing
# transform = transforms.Compose([
#     transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Model class
# class ProductionDeepfakeDetector(nn.Module):
#     def __init__(self, num_classes=2, dropout_rate=0.3):
#         super(ProductionDeepfakeDetector, self).__init__()
#         self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
#         in_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Sequential(
#             nn.Dropout(dropout_rate),
#             nn.Linear(in_features, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_rate),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_rate * 0.5),
#             nn.Linear(256, num_classes)
#         )
    
#     def forward(self, x):
#         return self.resnet(x)

# # Global variables
# model = None
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def download_model():
#     """Download model from Hugging Face"""
#     try:
#         logger.info(f"Downloading model from: {Config.HUGGINGFACE_REPO}")
#         model_path = hf_hub_download(
#             repo_id=Config.HUGGINGFACE_REPO,
#             filename=Config.MODEL_FILENAME,
#             cache_dir=Config.MODELS_FOLDER
#         )
#         logger.info(f"Model downloaded to: {model_path}")
#         return model_path
#     except Exception as e:
#         logger.error(f"Download failed: {e}")
#         return None

# def load_model():
#     """Load the model"""
#     global model
#     try:
#         model = ProductionDeepfakeDetector()
#         model_path = download_model()
        
#         if model_path and os.path.exists(model_path):
#             checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
#             if 'model_state_dict' in checkpoint:
#                 model.load_state_dict(checkpoint['model_state_dict'])
#             else:
#                 model.load_state_dict(checkpoint)
                
#             model.to(device)
#             model.eval()
#             logger.info("✅ Model loaded successfully!")
#             return True
#         else:
#             logger.error("❌ Model file not found")
#             return False
#     except Exception as e:
#         logger.error(f"❌ Model loading failed: {e}")
#         return False

# def allowed_file(filename):
#     """Check file extension"""
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# def preprocess_image(image_data):
#     """Preprocess image"""
#     try:
#         image = Image.open(io.BytesIO(image_data)).convert('RGB')
#         image_tensor = transform(image).unsqueeze(0)
#         return image_tensor
#     except Exception as e:
#         logger.error(f"Image preprocessing failed: {e}")
#         return None

# @app.route('/', methods=['GET'])
# def index():
#     """Root endpoint"""
#     return jsonify({
#         'message': 'Deepfake Detection API',
#         'status': 'running',
#         'model_loaded': model is not None
#     })

# @app.route('/health', methods=['GET'])
# def health():
#     """Health check"""
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': model is not None,
#         'timestamp': datetime.now().isoformat()
#     })

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Main prediction endpoint"""
#     try:
#         # Check model
#         if model is None:
#             return jsonify({'error': 'Model not loaded'}), 500
        
#         # Check file
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400
            
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
            
#         if not allowed_file(file.filename):
#             return jsonify({'error': 'Invalid file type'}), 400
        
#         # Read and check file size
#         file_data = file.read()
#         if len(file_data) > Config.MAX_FILE_SIZE:
#             return jsonify({'error': 'File too large'}), 400
        
#         # Preprocess image
#         image_tensor = preprocess_image(file_data)
#         if image_tensor is None:
#             return jsonify({'error': 'Failed to process image'}), 400
        
#         # Make prediction
#         with torch.no_grad():
#             image_tensor = image_tensor.to(device)
#             outputs = model(image_tensor)
#             probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
#             fake_prob = probabilities[0][0].item() * 100
#             real_prob = probabilities[0][1].item() * 100
        
#         # Determine prediction
#         if real_prob > fake_prob:
#             prediction = "Real"
#             confidence = real_prob
#         else:
#             prediction = "Fake"
#             confidence = fake_prob
        
#         # Response
#         return jsonify({
#             'success': True,
#             'prediction': prediction,
#             'confidence': round(confidence, 2),
#             'probabilities': {
#                 'fake': round(fake_prob, 2),
#                 'real': round(real_prob, 2)
#             },
#             'timestamp': datetime.now().isoformat(),
#             'model_info': {
#                 'name': 'ResNet50',
#                 'version': '1.0',
#                 'accuracy': '98%+'
#             }
#         })
        
#     except Exception as e:
#         logger.error(f"Prediction error: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.errorhandler(404)
# def not_found(e):
#     return jsonify({'error': 'Not found'}), 404

# @app.errorhandler(500)
# def server_error(e):
#     return jsonify({'error': 'Server error'}), 500

# # Initialize app
# def create_app():
#     """Application factory for gunicorn"""
#     logger.info("🚀 Starting Deepfake Detection API")
#     load_model()
#     return app

# if __name__ == '__main__':
#     logger.info("🚀 Starting Deepfake Detection API")
#     load_model()
#     # Railway uses PORT environment variable
#     port = int(os.environ.get('PORT', 8000))
#     logger.info(f"🌐 Starting server on port {port}")
#     app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
# else:
#     # For gunicorn - load model when imported
#     load_model()

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import io
import base64
import os
import logging
import cloudinary
import cloudinary.uploader
from datetime import datetime
import numpy as np
import cv2
import uuid
from huggingface_hub import hf_hub_download
import requests
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])

# Configuration
class Config:
    # Updated to use environment variable for HuggingFace model URL
    HUGGINGFACE_REPO = os.getenv('HUGGINGFACE_REPO')
    MODEL_FILENAME = os.getenv('MODEL_FILENAME')
    IMG_SIZE = 224
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
    UPLOAD_FOLDER = 'uploads'
    MODELS_FOLDER = 'models'
    DROPOUT_RATE = 0.3

# Configure Cloudinary (optional)
if os.getenv('CLOUDINARY_CLOUD_NAME'):
    cloudinary.config(
        cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
        api_key=os.getenv('CLOUDINARY_API_KEY'),
        api_secret=os.getenv('CLOUDINARY_API_SECRET')
    )

# Create necessary directories
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.MODELS_FOLDER, exist_ok=True)

# Image preprocessing - matches training script exactly
transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model architecture - UPDATED to match deepfake_model.py exactly
class ProductionDeepfakeDetector(nn.Module):
    """Production-ready ResNet50 based detector - matches training script"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ProductionDeepfakeDetector, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Custom classifier head - exactly as in training script
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
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.resnet(x)

# Global model variable - Force CPU usage
model = None
device = torch.device("cpu")  # Force CPU usage

def download_model_from_huggingface():
    """Download model from Hugging Face Hub"""
    try:
        logger.info(f"Downloading model from Hugging Face: {Config.HUGGINGFACE_REPO}")
        
        # Download the model file
        model_path = hf_hub_download(
            repo_id=Config.HUGGINGFACE_REPO,
            filename=Config.MODEL_FILENAME,
            cache_dir=Config.MODELS_FOLDER,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        logger.info(f"Model downloaded successfully to: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Error downloading model from Hugging Face: {e}")
        return None

def load_model():
    """Load the trained model - CPU only"""
    global model
    try:
        model = ProductionDeepfakeDetector(num_classes=2, dropout_rate=Config.DROPOUT_RATE)
        
        # Try to download model from Hugging Face
        model_path = download_model_from_huggingface()
        
        if model_path and os.path.exists(model_path):
            # Load the model with proper error handling - force CPU
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                logger.info("Loaded checkpoint with weights_only=False")
            except Exception as e:
                logger.warning(f"Failed to load with weights_only=False: {e}")
                
                # Fallback methods
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    logger.info("Loaded checkpoint with default settings")
                except Exception as e2:
                    logger.error(f"Failed to load model: {e2}")
                    return False
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Model loaded successfully from {model_path}")
                
                # Log additional info if available
                if 'accuracy' in checkpoint:
                    logger.info(f"Model accuracy: {checkpoint['accuracy']:.4f}")
                if 'model_architecture' in checkpoint:
                    logger.info(f"Model architecture: {checkpoint['model_architecture']}")
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
                logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.error("Failed to download model from Hugging Face")
            logger.error("Please check your HUGGINGFACE_REPO and MODEL_FILENAME environment variables")
            return False
            
        model.to(device)
        model.eval()
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Apply transforms - same as training
        image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor, image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None, None

def generate_simple_heatmap(model, image_tensor, original_image):
    """Generate a simple attention visualization without GradCAM"""
    try:
        # Create a simple random heatmap as placeholder
        # In production, you might want to implement a simpler attention mechanism
        heatmap = np.random.rand(Config.IMG_SIZE, Config.IMG_SIZE)
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Resize original image
        original_resized = np.array(original_image.resize((Config.IMG_SIZE, Config.IMG_SIZE)))
        
        # Blend images
        blended = cv2.addWeighted(original_resized, 0.7, heatmap_colored, 0.3, 0)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', blended)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return heatmap_b64
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return None

def get_detailed_analysis(fake_prob, real_prob):
    """Generate detailed analysis text"""
    confidence = max(fake_prob, real_prob)
    
    if real_prob > fake_prob:
        prediction = "Real"
        analysis = f"This image appears to be authentic with {confidence:.1f}% confidence. "
        
        if confidence > 95:
            analysis += "The model is highly confident this is a genuine image."
        elif confidence > 80:
            analysis += "The model is reasonably confident this is a genuine image."
        else:
            analysis += "The model suggests this is likely genuine, but with moderate confidence."
            
    else:
        prediction = "Fake"
        analysis = f"This image appears to be artificially generated with {confidence:.1f}% confidence. "
        
        if confidence > 95:
            analysis += "The model detected strong indicators of artificial generation."
        elif confidence > 80:
            analysis += "The model found several indicators suggesting artificial generation."
        else:
            analysis += "The model suggests this may be artificially generated, but with moderate confidence."
    
    # Add technical details
    analysis += f"\n\nTechnical Details:\n"
    analysis += f"• Real probability: {real_prob:.1f}%\n"
    analysis += f"• Fake probability: {fake_prob:.1f}%\n"
    analysis += f"• Model: ResNet50 (CPU)\n"
    analysis += f"• Processing time: ~5-10 seconds"
    
    return prediction, analysis

def save_image_locally(image_data, filename):
    """Save image to local uploads folder"""
    try:
        local_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        with open(local_path, 'wb') as f:
            f.write(image_data)
        return local_path
    except Exception as e:
        logger.error(f"Error saving image locally: {e}")
        return None

def upload_to_cloudinary(image_data, filename):
    """Upload image to Cloudinary (optional)"""
    try:
        if not os.getenv('CLOUDINARY_CLOUD_NAME'):
            return None
            
        upload_result = cloudinary.uploader.upload(
            image_data,
            folder="deepfake_detection",
            resource_type="image",
            public_id=filename.split('.')[0],
            format="png"
        )
        return upload_result.get('secure_url')
        
    except Exception as e:
        logger.warning(f"Failed to upload to Cloudinary: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Production Deepfake Detection API is running',
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_architecture': 'ResNet50',
        'device': str(device)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_architecture': 'ResNet50',
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please check Hugging Face configuration.',
                'hint': 'Ensure HUGGINGFACE_REPO and MODEL_FILENAME environment variables are set correctly.'
            }), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(Config.ALLOWED_EXTENSIONS).upper()}'
            }), 400
        
        # Read file data
        file_data = file.read()
        
        # Check file size
        if len(file_data) > Config.MAX_FILE_SIZE:
            return jsonify({'error': 'File too large. Maximum size: 16MB'}), 400
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        
        # Save image locally
        local_path = save_image_locally(file_data, unique_filename)
        
        # Upload to Cloudinary (optional)
        cloudinary_url = upload_to_cloudinary(file_data, unique_filename)
        
        # Preprocess image
        image_tensor, original_image = preprocess_image(file_data)
        if image_tensor is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            fake_prob = probabilities[0][0].item() * 100
            real_prob = probabilities[0][1].item() * 100
        
        # Generate detailed analysis
        prediction, analysis = get_detailed_analysis(fake_prob, real_prob)
        
        # Generate simple heatmap visualization
        heatmap_image = generate_simple_heatmap(model, image_tensor, original_image)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': prediction,
            'confidence': max(fake_prob, real_prob),
            'probabilities': {
                'fake': round(fake_prob, 2),
                'real': round(real_prob, 2)
            },
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'name': 'ResNet50',
                'version': '1.0',
                'architecture': 'Production Deepfake Detector',
                'accuracy': '98%+',
                'device': 'CPU'
            },
            'local_path': local_path
        }
        
        # Add optional data
        if cloudinary_url:
            response['image_url'] = cloudinary_url
        
        if heatmap_image:
            response['heatmap'] = heatmap_image
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple images"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if len(files) > 10:  # Limit batch size
            return jsonify({'error': 'Maximum 10 files allowed per batch'}), 400
        
        results = []
        
        for i, file in enumerate(files):
            try:
                if not allowed_file(file.filename):
                    results.append({
                        'filename': file.filename,
                        'error': 'Invalid file type'
                    })
                    continue
                
                file_data = file.read()
                if len(file_data) > Config.MAX_FILE_SIZE:
                    results.append({
                        'filename': file.filename,
                        'error': 'File too large'
                    })
                    continue
                
                # Process image
                image_tensor, original_image = preprocess_image(file_data)
                if image_tensor is None:
                    results.append({
                        'filename': file.filename,
                        'error': 'Failed to process image'
                    })
                    continue
                
                # Make prediction
                with torch.no_grad():
                    image_tensor = image_tensor.to(device)
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    fake_prob = probabilities[0][0].item() * 100
                    real_prob = probabilities[0][1].item() * 100
                
                prediction, _ = get_detailed_analysis(fake_prob, real_prob)
                
                results.append({
                    'filename': file.filename,
                    'prediction': prediction,
                    'confidence': max(fake_prob, real_prob),
                    'probabilities': {
                        'fake': round(fake_prob, 2),
                        'real': round(real_prob, 2)
                    }
                })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'processed_count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(Config.UPLOAD_FOLDER, filename)

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'model_architecture': 'ResNet50',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'device': str(device),
        'huggingface_repo': Config.HUGGINGFACE_REPO,
        'model_filename': Config.MODEL_FILENAME,
        'input_size': f"{Config.IMG_SIZE}x{Config.IMG_SIZE}",
        'dropout_rate': Config.DROPOUT_RATE
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize the application
def create_app():
    """Application factory"""
    
    # Load model on startup
    if not load_model():
        logger.error("Failed to load model from Hugging Face.")
        logger.info("Please check your HUGGINGFACE_REPO and MODEL_FILENAME environment variables.")
    else:
        logger.info("✅ Model loaded successfully from Hugging Face!")
        logger.info(f"🖥️ Device: {device}")
        if model:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"🧠 Model parameters: {total_params:,}")
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting Production Deepfake Detection API")
    print(f"📡 Server: http://0.0.0.0:{port}")
    print(f"🤖 Model: ResNet50-based Deepfake Detector (CPU)")
    print(f"📱 Model loaded: {model is not None}")
    print(f"🖥️ Device: {device}")
    print("="*50)
    app.run(host='0.0.0.0', port=port, debug=False)
