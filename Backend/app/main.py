# app/main.py - Windows-Optimized SecurePay Backend
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path (Windows path handling)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import your API routes (we'll create these)
# from app.api import auth, biometric, payments, users
# from app.core.config import settings
# from app.core.database import engine, Base

# Configure logging for Windows
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "securepay.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting SecurePay Backend Server on Windows...")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Create data directories if they don't exist
    data_dirs = ["data/voice_samples", "data/face_samples", "models"]
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize database tables
    # Base.metadata.create_all(bind=engine)
    
    yield
    
    # Shutdown
    logger.info("Shutting down SecurePay Backend Server...")

# Create FastAPI application
app = FastAPI(
    title="SecurePay with Voice + Face API",
    description="Revolutionary Multi-Modal UPI Payment System Backend - Windows Edition",
    version="1.0.0-windows",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React app
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "SecurePay with Voice + Face API - Windows Edition",
        "version": "1.0.0-windows",
        "status": "active",
        "platform": "Windows",
        "python_version": sys.version,
        "features": [
            "Voice Biometric Authentication",
            "Face Recognition with Liveness Detection",
            "Multi-modal Payment Processing",
            "Regional Language Support (Hindi, Tamil, Telugu)",
            "UPI Integration",
            "Intelligent Fallback System",
            "Windows-Optimized Audio/Video Processing"
        ],
        "api_docs": "/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    import platform
    return {
        "status": "healthy",
        "platform": {
            "system": platform.system(),
            "version": platform.version(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        },
        "services": {
            "database": "pending_setup",
            "redis": "pending_setup", 
            "voice_engine": "pending_setup",
            "face_engine": "pending_setup"
        },
        "directories": {
            "voice_samples": str(Path("data/voice_samples").exists()),
            "face_samples": str(Path("data/face_samples").exists()),
            "models": str(Path("models").exists())
        }
    }

# Test endpoints for development

@app.get("/api/v1/test/audio")
async def test_audio_support():
    """Test Windows audio support"""
    try:
        import soundfile as sf
        import librosa
        return {
            "status": "success",
            "audio_libraries": {
                "soundfile": sf.__version__,
                "librosa": librosa.__version__,
            },
            "supported_formats": [".wav", ".mp3", ".flac", ".m4a"]
        }
    except ImportError as e:
        return {
            "status": "error",
            "message": f"Audio library not available: {str(e)}"
        }

@app.get("/api/v1/test/vision")
async def test_vision_support():
    """Test Windows computer vision support"""
    try:
        import cv2
        import mediapipe as mp
        return {
            "status": "success",
            "vision_libraries": {
                "opencv": cv2.__version__,
                "mediapipe": mp.__version__,
            },
            "camera_support": "available"
        }
    except ImportError as e:
        return {
            "status": "error", 
            "message": f"Vision library not available: {str(e)}"
        }

# Voice processing endpoint (placeholder)
@app.post("/api/v1/voice/process")
async def process_voice_command(
    audio_file: UploadFile = File(...),
    language: str = "en"
):
    """Process voice command for payment intent"""
    try:
        # Validate file type
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an audio file."
            )
        
        # Save uploaded file temporarily (Windows path handling)
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"voice_{audio_file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Process audio file (placeholder logic)
        file_size = len(content)
        
        # Clean up temp file
        temp_file_path.unlink(missing_ok=True)
        
        return {
            "status": "success",
            "file_info": {
                "filename": audio_file.filename,
                "size_bytes": file_size,
                "content_type": audio_file.content_type
            },
            "processing_result": {
                "transcript": "Send 500 rupees to Ramesh",  # Placeholder
                "intent": "payment",
                "amount": 500,
                "recipient": "Ramesh",
                "confidence": 0.95,
                "language": language,
                "biometric_verified": False
            }
        }
    except Exception as e:
        logger.error(f"Voice processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

# Face verification endpoint (placeholder)
@app.post("/api/v1/face/verify")
async def verify_face(
    image_file: UploadFile = File(...),
    user_id: str = "demo_user"
):
    """Verify face against stored biometric"""
    try:
        # Validate file type
        if not image_file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file."
            )
        
        # Save uploaded file temporarily
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"face_{image_file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await image_file.read()
            buffer.write(content)
        
        file_size = len(content)
        
        # Clean up temp file
        temp_file_path.unlink(missing_ok=True)
        
        return {
            "status": "success",
            "file_info": {
                "filename": image_file.filename,
                "size_bytes": file_size,
                "content_type": image_file.content_type
            },
            "verification_result": {
                "verified": True,  # Placeholder
                "confidence": 0.97,
                "liveness_check": True,
                "face_detected": True,
                "user_id": user_id
            }
        }
    except Exception as e:
        logger.error(f"Face verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")

# Multi-modal authentication endpoint
@app.post("/api/v1/auth/multimodal")
async def multimodal_authentication(
    voice_file: UploadFile = File(None),
    face_file: UploadFile = File(None),
    user_id: str = "demo_user",
    fallback_pin: str = None
):
    """
    Multi-modal authentication with intelligent switching
    Voice -> Face -> PIN fallback
    """
    try:
        auth_result = {
            "user_id": user_id,
            "authenticated": False,
            "method_used": None,
            "confidence": 0.0,
            "fallback_available": [],
            "processing_details": []
        }
        
        # Try voice first if provided
        if voice_file and voice_file.filename:
            auth_result["processing_details"].append("Attempting voice authentication...")
            # Voice authentication logic here (placeholder)
            voice_confidence = 0.95
            if voice_confidence > 0.9:
                auth_result.update({
                    "authenticated": True,
                    "method_used": "voice",
                    "confidence": voice_confidence
                })
                auth_result["processing_details"].append("Voice authentication successful!")
                return auth_result
            else:
                auth_result["fallback_available"].append("face")
                auth_result["processing_details"].append("Voice authentication failed, trying face...")
        
        # Try face if voice failed or not provided
        if face_file and face_file.filename:
            auth_result["processing_details"].append("Attempting face authentication...")
            # Face authentication logic here (placeholder)
            face_confidence = 0.97
            if face_confidence > 0.9:
                auth_result.update({
                    "authenticated": True,
                    "method_used": "face",
                    "confidence": face_confidence
                })
                auth_result["processing_details"].append("Face authentication successful!")
                return auth_result
            else:
                auth_result["fallback_available"].append("pin")
                auth_result["processing_details"].append("Face authentication failed, PIN required...")
        
        # PIN fallback
        if fallback_pin:
            auth_result["processing_details"].append("Attempting PIN authentication...")
            # PIN verification logic here (placeholder)
            pin_valid = fallback_pin == "1234"  # Demo PIN
            if pin_valid:
                auth_result.update({
                    "authenticated": True,
                    "method_used": "pin",
                    "confidence": 1.0
                })
                auth_result["processing_details"].append("PIN authentication successful!")
                return auth_result
        
        # All methods failed
        auth_result["fallback_available"] = ["pin", "sms_otp"]
        auth_result["processing_details"].append("All authentication methods failed")
        return auth_result
        
    except Exception as e:
        logger.error(f"Multi-modal authentication error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

# Windows-specific utility endpoints
@app.get("/api/v1/system/info")
async def get_system_info():
    """Get Windows system information"""
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total": psutil.disk_usage('.').total,
            "free": psutil.disk_usage('.').free,
            "percent": (psutil.disk_usage('.').used / psutil.disk_usage('.').total) * 100
        }
    }

if __name__ == "__main__":
    print("Starting SecurePay Backend on Windows...")
    print(f"API Documentation available at: http://localhost:8000/docs")
    print(f"Interactive API at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",  # Use 127.0.0.1 instead of 0.0.0.0 for Windows
        port=8000,
        reload=True,
        log_level="info"
    )