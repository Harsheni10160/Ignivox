#!/usr/bin/env python3
"""
Startup script for SecurePay Backend
Checks dependencies and starts the server with proper error handling
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'librosa',
        'numpy',
        'speech_recognition',
        'scikit-learn',
        'sqlalchemy',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def start_server():
    """Start the FastAPI server"""
    try:
        print("🚀 Starting SecurePay Backend Server...")
        print("📍 Server will be available at: http://localhost:8000")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🔧 Health Check: http://localhost:8000/health")
        print("🎤 Voice Status: http://localhost:8000/voice/status")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    if check_dependencies():
        start_server()
    else:
        sys.exit(1)
