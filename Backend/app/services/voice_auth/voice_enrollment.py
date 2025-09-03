import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import pickle
import hashlib

class VoiceEnrollment:
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 3  # seconds
        self.scaler = StandardScaler()
    
    def extract_voice_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract voice features for biometric authentication
        Using MFCC (Mel-frequency cepstral coefficients) for voice recognition
        """
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=self.sample_rate, 
            n_mfcc=13
        )
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, 
            sr=self.sample_rate
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, 
            sr=self.sample_rate
        )
        
        # Extract rhythm features
        tempo, _ = librosa.beat.beat_track(
            y=audio_data, 
            sr=self.sample_rate
        )
        
        # Combine features
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            [np.mean(spectral_centroids)],
            [np.std(spectral_centroids)],
            [np.mean(spectral_rolloff)],
            [np.std(spectral_rolloff)],
            [tempo]
        ])
        
        return features
    
    def create_voice_template(self, audio_samples: List[np.ndarray]) -> str:
        """
        Create a voice template from multiple audio samples
        Returns encrypted template string
        """
        feature_vectors = []
        
        for audio in audio_samples:
            features = self.extract_voice_features(audio)
            feature_vectors.append(features)
        
        # Create average template
        template = np.mean(feature_vectors, axis=0)
        
        # Normalize features
        template_normalized = self.scaler.fit_transform(template.reshape(1, -1))
        
        # Convert to encrypted string
        template_bytes = pickle.dumps({
            'template': template_normalized,
            'scaler': self.scaler,
            'samples_count': len(audio_samples)
        })
        
        # Simple encryption (use proper encryption in production)
        template_hash = hashlib.sha256(template_bytes).hexdigest()
        
        return template_hash[:64]  # Return first 64 chars as template ID
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for better feature extraction
        """
        # Normalize audio
        audio_normalized = librosa.util.normalize(audio_data)
        
        # Remove silence
        audio_trimmed, _ = librosa.effects.trim(
            audio_normalized, 
            top_db=20
        )
        
        # Ensure consistent length
        if len(audio_trimmed) > self.sample_rate * self.duration:
            audio_trimmed = audio_trimmed[:self.sample_rate * self.duration]
        else:
            # Pad with zeros if too short
            padding = self.sample_rate * self.duration - len(audio_trimmed)
            audio_trimmed = np.pad(audio_trimmed, (0, padding), 'constant')
        
        return audio_trimmed
    
    def validate_audio_quality(self, audio_data: np.ndarray) -> Tuple[bool, str]:
        """
        Validate if audio is suitable for enrollment
        """
        # Check audio length
        if len(audio_data) < self.sample_rate * 2:  # At least 2 seconds
            return False, "Audio too short. Please record at least 2 seconds."
        
        # Check if audio has sufficient volume
        rms_energy = np.sqrt(np.mean(audio_data**2))
        if rms_energy < 0.01:
            return False, "Audio too quiet. Please speak louder."
        
        # Check for clipping
        if np.max(np.abs(audio_data)) > 0.95:
            return False, "Audio is clipped. Please reduce volume."
        
        # Basic silence detection
        silent_frames = np.sum(np.abs(audio_data) < 0.01) / len(audio_data)
        if silent_frames > 0.8:
            return False, "Too much silence detected. Please speak clearly."
        
        return True, "Audio quality is good."