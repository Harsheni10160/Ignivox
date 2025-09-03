from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.services.voice_auth.voice_enrollment import VoiceEnrollment
from app.services.voice_auth.voice_verification import VoiceVerification
from app.services.voice_auth.speech_intent_service import VoicePaymentService, PaymentIntent
from app.models.user import User
import numpy as np
import librosa
from typing import Dict, Any
from pydantic import BaseModel
import json
import io
import random
import soundfile as sf
import wave
import tempfile
import os
from app.utils.audio_converter import robust_audio_processing

router = APIRouter(prefix="/voice", tags=["voice-authentication"])

# Initialize services
voice_enrollment = VoiceEnrollment()
voice_verification = VoiceVerification()
voice_payment_service = VoicePaymentService()

class VoiceEnrollmentResponse(BaseModel):
    success: bool
    message: str
    template_id: str = None

class VoiceVerificationResponse(BaseModel):
    verified: bool
    confidence: float
    message: str

class VoicePaymentResponse(BaseModel):
    intent: Dict[str, Any]
    confirmation_text: str
    success: bool

def process_audio_file(audio_content: bytes) -> tuple[np.ndarray, int]:
    """
    Process audio file in various formats and convert to numpy array
    """
    try:
        # Use robust audio processing that tries multiple methods
        audio_data, sr = robust_audio_processing(audio_content)
        return audio_data, sr
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to process audio. Please ensure audio is in a supported format (WAV, MP3, WebM). Error: {str(e)}"
        )

@router.post("/enroll", response_model=VoiceEnrollmentResponse)
async def enroll_voice(
    user_id: int,
    audio_files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Enroll user's voice with multiple audio samples
    """
    try:
        if len(audio_files) < 3:
            raise HTTPException(
                status_code=400, 
                detail="At least 3 audio samples required for enrollment"
            )
        
        audio_samples = []
        
        for audio_file in audio_files:
            # Read audio file
            audio_content = await audio_file.read()
            
            # Convert to numpy array (assuming WAV format)
            audio_data, sr = process_audio_file(audio_content)
            
            # Validate audio quality
            is_valid, message = voice_enrollment.validate_audio_quality(audio_data)
            if not is_valid:
                return VoiceEnrollmentResponse(
                    success=False,
                    message=f"Audio quality issue: {message}"
                )
            
            # Preprocess audio
            processed_audio = voice_enrollment.preprocess_audio(audio_data)
            audio_samples.append(processed_audio)
        
        # Create voice template
        template_id = voice_enrollment.create_voice_template(audio_samples)

        # Store in database
        user: User | None = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        user.voice_template = template_id
        db.add(user)
        db.commit()
        
        return VoiceEnrollmentResponse(
            success=True,
            message="Voice enrolled successfully",
            template_id=template_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")

@router.post("/verify", response_model=VoiceVerificationResponse)
async def verify_voice(
    user_id: int,
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Verify user's voice against stored template
    """
    try:
        # Read audio file
        audio_content = await audio_file.read()
        audio_data, sr = process_audio_file(audio_content)
        
        # Get stored template from DB
        user: User | None = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        if not user.voice_template:
            return VoiceVerificationResponse(
                verified=False,
                confidence=0.0,
                message="No enrolled voice template for user"
            )
        stored_template = user.voice_template
        
        # Check liveness
        is_live, liveness_message = voice_verification.detect_liveness(audio_data)
        if not is_live:
            return VoiceVerificationResponse(
                verified=False,
                confidence=0.0,
                message=f"Liveness check failed: {liveness_message}"
            )
        
        # Check voice health
        is_healthy, health_message = voice_verification.check_voice_health(audio_data)
        if not is_healthy:
            return VoiceVerificationResponse(
                verified=False,
                confidence=0.0,
                message=health_message
            )
        
        # Verify voice
        is_verified, confidence = voice_verification.verify_voice(audio_data, stored_template)
        
        return VoiceVerificationResponse(
            verified=is_verified,
            confidence=confidence,
            message="Voice verification successful" if is_verified else "Voice verification failed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@router.post("/payment", response_model=VoicePaymentResponse)
async def process_voice_payment(
    user_id: int,
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Complete voice payment flow: Verification + Intent Recognition + Payment Processing
    """
    try:
        # Read audio file
        audio_content = await audio_file.read()
        audio_data, sr = process_audio_file(audio_content)
        
        # Step 1: Voice Verification
        user: User | None = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        if not user.voice_template:
            return VoicePaymentResponse(
                intent={},
                confirmation_text="No enrolled voice template for user",
                success=False
            )
        stored_template = user.voice_template
        is_verified, confidence = voice_verification.verify_voice(audio_data, stored_template)
        
        if not is_verified:
            return VoicePaymentResponse(
                intent={},
                confirmation_text="Voice verification failed. Please try again.",
                success=False
            )
        
        # Step 2: Process Voice Command
        payment_intent, confirmation_text = voice_payment_service.process_voice_command(audio_data)
        
        if payment_intent.action == 'error' or payment_intent.action == 'unknown':
            return VoicePaymentResponse(
                intent=payment_intent.__dict__,
                confirmation_text=confirmation_text,
                success=False
            )
        
        # Step 3: Process Payment (mock implementation)
        success = True
        if payment_intent.action == 'send_money':
            # Here you would integrate with actual UPI/payment gateway
            # For now, we'll simulate success
            success = await simulate_payment_processing(payment_intent)
            
            if success:
                confirmation_text = f"Successfully sent ₹{payment_intent.amount} to {payment_intent.recipient}"
            else:
                confirmation_text = "Payment failed. Please try again."
                
        return VoicePaymentResponse(
            intent=payment_intent.__dict__,
            confirmation_text=confirmation_text,
            success=success if payment_intent.action == 'send_money' else True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Payment processing failed: {str(e)}")

@router.post("/command")
async def process_voice_command_only(
    audio_file: UploadFile = File(...)
):
    """
    Process voice command without authentication (for testing)
    """
    try:
        # Validate file
        if not audio_file:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        # Read audio file
        audio_content = await audio_file.read()
        
        if len(audio_content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Convert to numpy array
        try:
            audio_data, sr = process_audio_file(audio_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load audio: {str(e)}")
        
        # Process command
        try:
            payment_intent, confirmation_text = voice_payment_service.process_voice_command(audio_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")
        
        return {
            "intent": payment_intent.__dict__,
            "confirmation": confirmation_text,
            "success": payment_intent.action != 'unknown',
            "recognized_text": confirmation_text,
            "audio_info": {
                "duration": len(audio_data) / sr,
                "sample_rate": sr,
                "file_size": len(audio_content)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command processing failed: {str(e)}")

@router.post("/debug-speech")
async def debug_speech_recognition(
    audio_file: UploadFile = File(...)
):
    """
    Debug endpoint to test speech recognition separately
    """
    try:
        # Read audio file
        audio_content = await audio_file.read()
        
        # Convert to numpy array
        audio_data, sr = process_audio_file(audio_content)
        
        # Test speech recognition directly
        from app.services.voice_auth.speech_intent_service import SpeechToTextService
        speech_service = SpeechToTextService()
        
        # Try different languages
        results = {}
        
        for lang_name, lang_code in speech_service.supported_languages.items():
            try:
                text, confidence = speech_service.convert_audio_to_text(audio_data, lang_code)
                results[lang_name] = {
                    "text": text,
                    "confidence": confidence,
                    "language_code": lang_code
                }
            except Exception as e:
                results[lang_name] = {
                    "error": str(e),
                    "confidence": 0.0
                }
        
        return {
            "audio_info": {
                "duration": len(audio_data) / sr,
                "sample_rate": sr,
                "file_size": len(audio_content)
            },
            "speech_results": results,
            "best_result": max(results.items(), key=lambda x: x[1].get('confidence', 0))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

async def simulate_payment_processing(intent: PaymentIntent) -> bool:
    """
    Simulate payment processing
    In production, this would integrate with UPI/payment gateway
    """
    # Simulate some basic validation
    if intent.amount <= 0:
        return False
    
    if intent.amount > 50000:  # Daily limit check
        return False
    
    if not intent.recipient:
        return False
    
    # Simulate 90% success rate
    return random.random() > 0.1