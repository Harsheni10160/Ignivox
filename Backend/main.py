from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.auth import router as auth_router
from app.api.websocket import router as websocket_router
from app.api.voice import router as voice_router
# from app.api.face import router as face_router

app = FastAPI(
    title="SecurePay API",
    description="Revolutionary Multi-Modal UPI Payment System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(websocket_router)
app.include_router(voice_router)
# app.include_router(face_router)

@app.get("/")
async def root():
    return {
        "message": "SecurePay API is running!",
        "features": [
            "Voice Authentication",
            "Speech-to-Text Processing",
            "Intent Recognition",
            "Voice Payment Processing"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/voice/test")
async def voice_test():
    return {
        "message": "Voice services are ready",
        "supported_commands": [
            "Send [amount] to [person]",
            "Pay [amount] to [person]",
            "Transfer [amount] to [person]",
            "Check my balance",
            "Hindi: [व्यक्ति] को [राशि] रुपये भेजो"
        ]
    }

@app.get("/voice/status")
async def voice_status():
    """Check if voice services are properly initialized"""
    try:
        from app.services.voice_auth.speech_intent_service import VoicePaymentService
        service = VoicePaymentService()
        return {
            "status": "ready",
            "message": "Voice services initialized successfully",
            "services": ["speech_to_text", "intent_recognition", "voice_verification"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Voice services initialization failed: {str(e)}",
            "error": str(e)
        }