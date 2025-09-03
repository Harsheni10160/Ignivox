#!/usr/bin/env python3
"""
Test script for Voice API endpoints
"""

import requests
import numpy as np
import io
import wave
import tempfile
import os

def create_test_audio():
    """Create a simple test audio file"""
    # Generate a simple sine wave (1 second, 440 Hz)
    sample_rate = 16000
    duration = 1
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_file.name

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_voice_status():
    """Test the voice status endpoint"""
    print("\n🔍 Testing voice status endpoint...")
    try:
        response = requests.get("http://localhost:8000/voice/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Voice status endpoint working")
            print(f"   Status: {data.get('status')}")
            print(f"   Message: {data.get('message')}")
            return data.get('status') == 'ready'
        else:
            print(f"❌ Voice status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Voice status error: {e}")
        return False

def test_voice_command():
    """Test the voice command endpoint"""
    print("\n🔍 Testing voice command endpoint...")
    try:
        # Create test audio file
        audio_file = create_test_audio()
        
        with open(audio_file, 'rb') as f:
            files = {'audio_file': ('test.wav', f, 'audio/wav')}
            response = requests.post("http://localhost:8000/voice/command", files=files)
        
        # Clean up
        os.unlink(audio_file)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Voice command endpoint working")
            print(f"   Intent: {data.get('intent', {}).get('action')}")
            print(f"   Success: {data.get('success')}")
            return True
        else:
            print(f"❌ Voice command failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Voice command error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing SecurePay Voice API...")
    print("=" * 50)
    
    # Test health
    health_ok = test_health_endpoint()
    
    # Test voice status
    voice_ok = test_voice_status()
    
    # Test voice command
    command_ok = test_voice_command()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Health Endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   Voice Status: {'✅ PASS' if voice_ok else '❌ FAIL'}")
    print(f"   Voice Command: {'✅ PASS' if command_ok else '❌ FAIL'}")
    
    if all([health_ok, voice_ok, command_ok]):
        print("\n🎉 All tests passed! The API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the server logs for details.")

if __name__ == "__main__":
    main()
