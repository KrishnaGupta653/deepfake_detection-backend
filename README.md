# AI Deepfake Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-98%25+-brightgreen.svg)](/)

> Advanced AI-powered deepfake detection system using ResNet50 architecture with industry-leading accuracy of 98%+. Features real-time analysis, visual explanations through Grad-CAM, and a modern web interface.

## 🌟 Features

- **🧠 Advanced AI Detection**: ResNet50-based neural network trained on 500K+ images
- **⚡ Lightning Fast**: Sub-3 second processing with GPU acceleration
- **🎯 High Accuracy**: 98%+ detection rate with continuous model improvements
- **📊 Visual Explanations**: Grad-CAM heatmaps show AI decision-making process
- **🌐 Modern Web Interface**: Beautiful, responsive UI built with Next.js and Tailwind CSS
- **🔒 Secure & Private**: Images processed securely without server storage
- **📱 Mobile Friendly**: Works seamlessly across all devices
- **🚀 Production Ready**: Scalable architecture with comprehensive error handling

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │  AI Model       │
│   (Next.js)     │────│   (Flask API)    │────│  (PyTorch)      │
│                 │    │                  │    │                 │
│ • File Upload   │    │ • Image Processing│    │ • ResNet50      │
│ • Results UI    │    │ • Model Inference │    │ • Grad-CAM      │
│ • Visualizations│    │ • API Endpoints   │    │ • Custom Head   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔄 Processing Flow

```mermaid
graph TD
    A[User Uploads Image] --> B[Frontend Validation]
    B --> C[Send to Flask API]
    C --> D[Image Preprocessing]
    D --> E[ResNet50 Model Inference]
    E --> F[Generate Predictions]
    F --> G[Create Grad-CAM Visualization]
    G --> H[Return Results to Frontend]
    H --> I[Display Results & Analysis]
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style I fill:#e8f5e8
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 16.0 or higher
- **GPU**: CUDA-compatible GPU (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space

### 1. Clone Repository

```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HUGGINGFACE_REPO="your-username/deepfake-model"
export MODEL_FILENAME="production_deepfake_detector.pth"
export NEXT_PUBLIC_API_URL="http://localhost:8000"

# Run Flask server
python app.py
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend  # or wherever your Next.js app is located

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Access Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health

## 📊 Model Details

### Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Custom Head**: 3-layer classifier with batch normalization and dropout
- **Input Size**: 224×224 RGB images
- **Output**: Binary classification (Real/Fake)
- **Parameters**: 25M+ total parameters

### Training Specifications
- **Dataset**: 500K+ verified real and fake images
- **Training Split**: 80% train, 20% validation
- **Optimization**: AdamW with weight decay
- **Learning Rate**: 0.0001 with ReduceLROnPlateau scheduler
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Epochs**: 30 with early stopping

### Performance Metrics
```
Accuracy:  98.5%
Precision: 98.2%
Recall:    98.7%
F1-Score:  98.4%
```


### Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid file, missing parameters)
- `413`: Payload Too Large (file > 16MB)
- `500`: Internal Server Error

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Required
HUGGINGFACE_REPO=your-username/deepfake-model
MODEL_FILENAME=production_deepfake_detector.pth
NEXT_PUBLIC_API_URL=http://localhost:8000

# Optional - Cloudinary Integration
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret

# Optional - Hugging Face Token (for private models)
HUGGINGFACE_TOKEN=your-hf-token
```

### Model Configuration

Update `Config` class in `app.py`:

```python
class Config:
    IMG_SIZE = 224
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    DROPOUT_RATE = 0.3
```

## 🎨 Frontend Features

### Component Structure
```
src/
├── app/
│   ├── globals.css     # Tailwind CSS styles
│   ├── layout.tsx      # Root layout
│   └── page.tsx        # Main application
├── components/
│   ├── Navbar.tsx      # Navigation component
│   ├── UploadForm.tsx  # File upload interface
│   └── ResultDisplay.tsx # Results visualization
```

### Key Features
- **Drag & Drop Upload**: Intuitive file upload with validation
- **Real-time Processing**: Live feedback during analysis
- **Visual Results**: Confidence meters, probability charts
- **Grad-CAM Visualization**: AI attention heatmaps
- **Responsive Design**: Mobile-first approach
- **Dark Mode Support**: System preference detection
- **Export Functionality**: Download results as JSON

## 🧪 Training Your Own Model

### Dataset Structure
```
dataset/
├── real/           # Authentic images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── fake/           # Generated/manipulated images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Training Process

1. **Prepare Dataset**: Organize images in the structure above
2. **Update Paths**: Modify `Config` class in `deepfake_model.py`
3. **Start Training**:
   ```bash
   python deepfake_model.py
   ```
4. **Monitor Progress**: Training metrics displayed in real-time
5. **Model Evaluation**: Comprehensive performance analysis

### Training Configuration
```python
class Config:
    REAL_PATH = "dataset/real"
    FAKE_PATH = "dataset/fake"
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    IMG_SIZE = 224
```

## 📈 Performance Optimization

### GPU Acceleration
- Automatic CUDA detection and utilization
- Memory optimization with gradient clipping
- Efficient batch processing

### Model Optimizations
- Pre-trained ResNet50 backbone
- Custom classification head with dropout
- Batch normalization for stable training
- Learning rate scheduling

### Production Optimizations
- Model quantization support
- ONNX export capability
- TensorRT optimization (optional)
- Multi-GPU support

## 🐳 Docker Deployment

### Build Images
```bash
# Backend
docker build -t deepfake-api .

# Frontend
docker build -t deepfake-ui -f Dockerfile.frontend .
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HUGGINGFACE_REPO=your-repo
      - MODEL_FILENAME=model.pth
    volumes:
      - ./models:/app/models
  
  ui:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://api:8000
```

## 🚀 Deployment Options

### Cloud Platforms
- **Railway**: Easy deployment with automatic SSL
- **Heroku**: Simple git-based deployment
- **AWS**: EC2 instances with GPU support
- **Google Cloud**: Compute Engine with TPU options
- **Azure**: Container instances with ML services

### Deployment Checklist
- [ ] Environment variables configured
- [ ] Model files uploaded to Hugging Face
- [ ] SSL certificates configured
- [ ] Domain name pointed to server
- [ ] Health monitoring setup
- [ ] Error logging configured

