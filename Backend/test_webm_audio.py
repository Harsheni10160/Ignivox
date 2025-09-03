#!/usr/bin/env python3
"""
Test script to verify webm audio format support
"""

import requests
import numpy as np
import tempfile
import os
import wave

def create_webm_test_audio():
    """Create a test audio file that simulates webm format"""
    # Generate a simple sine wave
    sample_rate = 16000
    duration = 1
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create a file with webm-like header
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
        # Write a simple webm-like header (this is a simplified version)
        temp_file.write(b'\x1a\x45\xdf\xa3')  # WebM signature
        temp_file.write(audio_data.tobytes())
        
        return temp_file.name

def test_webm_audio_processing():
    """Test webm audio processing"""
    print("🔍 Testing webm audio format support...")
    
    try:
        # Create test webm file
        webm_file = create_webm_test_audio()
        
        with open(webm_file, 'rb') as f:
            files = {'audio_file': ('test.webm', f, 'audio/webm')}
            response = requests.post("http://localhost:8000/voice/command", files=files)
        
        # Clean up
        os.unlink(webm_file)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Webm audio processing successful")
            print(f"   Response: {data}")
            return True
        else:
            print(f"❌ Webm audio processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Webm audio test error: {e}")
        return False

def test_wav_audio_processing():
    """Test wav audio processing"""
    print("\n🔍 Testing wav audio format support...")
    
    try:
        # Create test wav file
        sample_rate = 16000
        duration = 1
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_file_path = temp_file.name
        
        with open(wav_file_path, 'rb') as f:
            files = {'audio_file': ('test.wav', f, 'audio/wav')}
            response = requests.post("http://localhost:8000/voice/command", files=files)
        
        # Clean up
        os.unlink(wav_file_path)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Wav audio processing successful")
            print(f"   Response: {data}")
            return True
        else:
            print(f"❌ Wav audio processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Wav audio test error: {e}")
        return False

def main():
    """Run audio format tests"""
    print("🎵 Testing Audio Format Support...")
    print("=" * 50)
    
    # Test webm format
    webm_ok = test_webm_audio_processing()
    
    # Test wav format
    wav_ok = test_wav_audio_processing()
    
    print("\n" + "=" * 50)
    print("📊 Audio Format Test Results:")
    print(f"   WebM Format: {'✅ PASS' if webm_ok else '❌ FAIL'}")
    print(f"   WAV Format: {'✅ PASS' if wav_ok else '❌ FAIL'}")
    
    if all([webm_ok, wav_ok]):
        print("\n🎉 All audio formats supported! Browser recordings should work.")
    else:
        print("\n⚠️ Some audio formats failed. Check the implementation.")

if __name__ == "__main__":
    main()
