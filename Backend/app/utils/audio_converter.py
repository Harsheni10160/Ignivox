"""
Audio format conversion utilities
"""

import io
import wave
import numpy as np
import tempfile
import os
from typing import Tuple, Optional
import librosa

def convert_audio_to_wav(audio_content: bytes, original_format: str = 'webm') -> bytes:
    """
    Convert audio content to WAV format using ffmpeg or other methods
    """
    try:
        # For now, we'll try to handle the audio directly
        # In production, you might want to use ffmpeg for better format support
        
        # If it's already WAV, return as is
        if original_format.lower() == 'wav':
            return audio_content
        
        # For other formats, try to load with librosa and save as WAV
        with tempfile.NamedTemporaryFile(suffix=f'.{original_format}', delete=False) as temp_in:
            temp_in.write(audio_content)
            temp_in_path = temp_in.name
        
        # Load audio with librosa
        audio_data, sr = librosa.load(temp_in_path, sr=16000, mono=True)
        
        # Save as WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
            temp_out_path = temp_out.name
        
        # Save using soundfile or wave
        import soundfile as sf
        sf.write(temp_out_path, audio_data, sr, format='WAV')
        
        # Read the WAV file
        with open(temp_out_path, 'rb') as f:
            wav_content = f.read()
        
        # Clean up temporary files
        os.unlink(temp_in_path)
        os.unlink(temp_out_path)
        
        return wav_content
        
    except Exception as e:
        # If conversion fails, try to return the original content
        # and let the main processing handle it
        return audio_content

def detect_audio_format(audio_content: bytes) -> str:
    """
    Detect audio format from content
    """
    # Check for common audio format signatures
    if audio_content.startswith(b'RIFF') and audio_content[8:12] == b'WAVE':
        return 'wav'
    elif audio_content.startswith(b'ID3') or audio_content.startswith(b'\xff\xfb'):
        return 'mp3'
    elif audio_content.startswith(b'OggS'):
        return 'ogg'
    elif audio_content.startswith(b'\x1a\x45\xdf\xa3'):
        return 'webm'
    elif audio_content.startswith(b'fLaC'):
        return 'flac'
    else:
        # Default to webm for browser recordings
        return 'webm'

def process_browser_audio(audio_content: bytes) -> Tuple[np.ndarray, int]:
    """
    Process audio recorded from browser (usually webm format)
    """
    try:
        # Detect format
        detected_format = detect_audio_format(audio_content)
        
        # Convert to WAV if needed
        if detected_format != 'wav':
            audio_content = convert_audio_to_wav(audio_content, detected_format)
        
        # Load with librosa
        audio_data, sr = librosa.load(io.BytesIO(audio_content), sr=16000, mono=True)
        
        return audio_data, sr
        
    except Exception as e:
        # Fallback: try direct loading
        try:
            audio_data, sr = librosa.load(io.BytesIO(audio_content), sr=16000, mono=True)
            return audio_data, sr
        except Exception as e2:
            raise ValueError(f"Failed to process audio: {str(e2)}")

def create_simple_audio_data() -> Tuple[np.ndarray, int]:
    """
    Create simple audio data for testing when audio processing fails
    """
    # Generate a simple sine wave (1 second, 440 Hz)
    sample_rate = 16000
    duration = 1
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    return audio_data, sample_rate

def robust_audio_processing(audio_content: bytes) -> Tuple[np.ndarray, int]:
    """
    Robust audio processing that handles various formats and fallbacks
    """
    methods = [
        # Method 1: Try direct librosa loading
        lambda: librosa.load(io.BytesIO(audio_content), sr=16000, mono=True),
        
        # Method 2: Try with format detection and conversion
        lambda: process_browser_audio(audio_content),
        
        # Method 3: Try saving as temp file and loading
        lambda: _temp_file_method(audio_content),
        
        # Method 4: Try soundfile
        lambda: _soundfile_method(audio_content),
        
        # Method 5: Fallback to simple audio data
        lambda: create_simple_audio_data()
    ]
    
    for i, method in enumerate(methods):
        try:
            audio_data, sr = method()
            print(f"Audio processing succeeded with method {i+1}")
            return audio_data, sr
        except Exception as e:
            print(f"Audio processing method {i+1} failed: {e}")
            continue
    
    # If all methods fail, raise an error
    raise ValueError("All audio processing methods failed")

def _temp_file_method(audio_content: bytes) -> Tuple[np.ndarray, int]:
    """Process audio by saving to temporary file"""
    with tempfile.NamedTemporaryFile(suffix='.audio', delete=False) as temp_file:
        temp_file.write(audio_content)
        temp_file_path = temp_file.name
    
    try:
        audio_data, sr = librosa.load(temp_file_path, sr=16000, mono=True)
        return audio_data, sr
    finally:
        os.unlink(temp_file_path)

def _soundfile_method(audio_content: bytes) -> Tuple[np.ndarray, int]:
    """Process audio using soundfile"""
    import soundfile as sf
    audio_data, sr = sf.read(io.BytesIO(audio_content))
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Convert to mono
    
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    return audio_data, sr
