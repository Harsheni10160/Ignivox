# SecurePay Backend - Voice API

This is the backend server for the SecurePay voice payment system. It provides voice authentication, speech-to-text processing, and voice command recognition.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

**Option A: Using the startup script (Recommended)**
```bash
python start_server.py
```

**Option B: Direct uvicorn command**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Test the API

- **Health Check**: http://localhost:8000/health
- **Voice Status**: http://localhost:8000/voice/status
- **API Documentation**: http://localhost:8000/docs
- **Voice Test Page**: Open `voice_test.html` in your browser

## Troubleshooting Common Issues

### 404 Error: "Failed to load resource: the server responded with a status of 404"

**Cause**: The server is not running or the route is not properly registered.

**Solution**:
1. Make sure the server is running on port 8000
2. Check if you can access http://localhost:8000/health
3. If the health endpoint works but voice endpoints don't, restart the server

### 500 Error: "Internal Server Error"

**Common Causes and Solutions**:

#### 1. Missing Dependencies
```bash
# Install missing packages
pip install SpeechRecognition pocketsphinx
```

#### 2. Audio Processing Issues
- Make sure you have a working microphone
- Record audio for at least 2-3 seconds
- Speak clearly and avoid background noise

#### 3. Speech Recognition Issues
The system uses Google Speech Recognition API. If it fails:
- Check your internet connection
- The system will fall back to offline recognition (pocketsphinx)

### Debugging Steps

1. **Check Server Status**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check Voice Services**:
   ```bash
   curl http://localhost:8000/voice/status
   ```

3. **View Server Logs**:
   The server will show detailed error messages in the console.

4. **Test with Simple Audio**:
   - Record a simple command like "send 100 to mom"
   - Make sure the audio is clear and not too short

## API Endpoints

### Voice Commands
- `POST /voice/command` - Process voice command (no authentication required for testing)
- `POST /voice/verify` - Verify user voice
- `POST /voice/enroll` - Enroll new voice
- `POST /voice/payment` - Complete voice payment flow

### Health & Status
- `GET /health` - Server health check
- `GET /voice/status` - Voice services status
- `GET /voice/test` - List supported commands

## Supported Voice Commands

- "Send [amount] to [person]"
- "Pay [amount] to [person]"
- "Transfer [amount] to [person]"
- "Check my balance"
- Hindi: "[व्यक्ति] को [राशि] रुपये भेजो"

## Development

### Project Structure
```
app/
├── api/
│   ├── voice.py          # Voice API endpoints
│   ├── auth.py           # Authentication endpoints
│   └── websocket.py      # WebSocket endpoints
├── services/
│   └── voice_auth/
│       ├── speech_intent_service.py    # Speech-to-text & intent recognition
│       ├── voice_enrollment.py         # Voice enrollment
│       └── voice_verification.py       # Voice verification
└── main.py               # FastAPI application
```

### Adding New Voice Commands

1. Edit `app/services/voice_auth/speech_intent_service.py`
2. Add new patterns to `payment_patterns` dictionary
3. Update the `get_confirmation_text` method

## Environment Variables

Create a `.env` file for configuration:
```env
DATABASE_URL=postgresql://user:password@localhost/securepay
SECRET_KEY=your-secret-key
```

## Dependencies

Key dependencies:
- `fastapi` - Web framework
- `librosa` - Audio processing
- `speech_recognition` - Speech-to-text
- `scikit-learn` - Machine learning utilities
- `numpy` - Numerical computing

## Support

If you encounter issues:
1. Check the server logs for detailed error messages
2. Verify all dependencies are installed
3. Test with the provided `voice_test.html` page
4. Check the API documentation at http://localhost:8000/docs
