import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from .voice_enrollment import VoiceEnrollment
import pickle
import hashlib
from typing import Tuple

class VoiceVerification:
    def __init__(self):
        self.enrollment = VoiceEnrollment()
        self.threshold = 0.85  # Similarity threshold for verification
        
    def verify_voice(self, audio_data: np.ndarray, stored_template: str) -> Tuple[bool, float]:
        """
        Verify voice against stored template
        Returns (is_verified, confidence_score)
        """
        try:
            # Preprocess audio
            processed_audio = self.enrollment.preprocess_audio(audio_data)
            
            # Extract features from current audio
            current_features = self.enrollment.extract_voice_features(processed_audio)
            
            # For demo purposes, we'll simulate template comparison
            # In production, you'd decrypt and compare with actual stored template
            confidence_score = self._simulate_template_comparison(current_features, stored_template)
            
            is_verified = confidence_score >= self.threshold
            
            return is_verified, confidence_score
            
        except Exception as e:
            print(f"Voice verification error: {e}")
            return False, 0.0
    
    def _simulate_template_comparison(self, features: np.ndarray, template: str) -> float:
        """
        Simulate template comparison for demo
        In production, this would compare actual feature vectors
        """
        # Create a mock comparison based on audio features
        feature_hash = hashlib.sha256(str(features).encode()).hexdigest()[:10]
        template_hash = template[:10]
        
        # Simple similarity simulation
        matches = sum(1 for a, b in zip(feature_hash, template_hash) if a == b)
        base_similarity = matches / 10
        
        # Add some randomness to simulate real comparison
        noise = np.random.normal(0, 0.1)
        similarity = max(0, min(1, base_similarity + 0.3 + noise))
        
        return similarity
    
    def detect_liveness(self, audio_data: np.ndarray) -> Tuple[bool, str]:
        """
        Basic liveness detection to prevent playback attacks
        """
        # Check for natural speech characteristics
        
        # 1. Check spectral variations (live speech has more variation)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(
            y=audio_data, 
            sr=self.enrollment.sample_rate
        ))
        
        spectral_variation = np.std(librosa.feature.spectral_centroid(
            y=audio_data, 
            sr=self.enrollment.sample_rate
        ))
        
        if spectral_variation < 50:  # Too uniform, might be playback
            return False, "Audio appears to be played back recording"
        
        # 2. Check for natural pauses and breathing
        rms_energy = librosa.feature.rms(y=audio_data)[0]
        energy_variation = np.std(rms_energy)
        
        if energy_variation < 0.01:  # Too uniform energy
            return False, "Audio lacks natural speech variations"
        
        # 3. Check frequency characteristics
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=self.enrollment.sample_rate, 
            n_mfcc=13
        )
        
        mfcc_variation = np.std(mfccs)
        if mfcc_variation < 5:
            return False, "Audio lacks natural voice characteristics"
        
        return True, "Liveness detected - appears to be live speech"
    
    def check_voice_health(self, audio_data: np.ndarray) -> Tuple[bool, str]:
        """
        Detect if user's voice might be affected by illness
        """
        # Extract features that might indicate voice issues
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, 
            sr=self.enrollment.sample_rate
        )
        
        # Check for hoarseness indicators
        avg_frequency = np.mean(spectral_centroids)
        
        # Hoarse voice typically has lower frequencies
        if avg_frequency < 800:  # Lower than typical speech
            return False, "Voice appears hoarse or affected. Consider using face authentication."
        
        # Check for breathiness
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, 
            sr=self.enrollment.sample_rate
        )
        
        avg_rolloff = np.mean(spectral_rolloff)
        if avg_rolloff < 2000:
            return False, "Voice quality affected. Switching to face authentication recommended."
        
        return True, "Voice sounds healthy"