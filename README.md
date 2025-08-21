## 🔐 SecurePay with Voice + Face [Ignivox]

Revolutionary Multi-Modal UPI Payment System for Rural & Elderly Users

#### 🚩 Problem

India’s UPI revolution excludes millions due to:

Rural users struggling with English keyboards & UPI IDs

Elderly citizens unable to handle OTPs & small screens

Regional language barriers

Voice-only solutions prone to fraud & noise failures

👉 60% of rural India still relies on cash despite UPI availability.

#### 💡 Solution

SecurePay with Voice + Face → India’s first dual biometric UPI system that adapts to YOU.

🎙 Voice Authentication (liveness detection, regional languages)

👤 Face Recognition (3D depth, anti-spoofing)

🔄 Smart Switching: Voice ⇆ Face ⇆ PIN fallback

🌐 Offline-First: Works on 2G, queues payments

🧓 Elder-Friendly: Simple UI, large buttons, natural language

#### 🦾 Key Features

Dual Biometric Security → Voice + Face + PIN fallback

Natural Language Payments → “Send ₹500 to Ramesh”

Multi-Language Support → Hindi, Tamil, Telugu, Kannada, Bengali

Accessibility First → Visually & hearing-impaired friendly

Anti-Fraud → Spoofing resistance, behavioral biometrics

Rural Ready → SMS/USSD fallback, 2G optimized

#### 🏗️ Tech Stack

Frontend: React Native, Expo, NativeBase

Backend: FastAPI (Python), WebSockets

Voice: Wav2Vec2, SpeechBrain, Whisper AI

Face: OpenCV, FaceNet, MediaPipe, ArcFace

NLP: Rasa, spaCy

UPI: NPCI APIs / Razorpay Sandbox

DB & Security: PostgreSQL, Redis, AES, JWT, bcrypt

🔄 Workflow

Enrollment → User records voice & face with liveness check

Payment → “Send ₹500 to Priya” → Voice/Face auth → UPI transfer

Fallbacks → Auto-switch to Face in noisy areas, to Voice in dark, to PIN if both fail

Confirmation → Voice/visual feedback + SMS backup

### 🎯 Hackathon Demo

✅ Voice → Payment succeeds

❌ Voice fail (cold/noise) → Face succeeds

❌ Spoof attempts (recorded voice/photo) → Blocked

🌐 Regional demo → Hindi + Face auth

🆘 Emergency → Silent face-only payment

📊 Success Metrics

⏱ <2s transaction speed

✅ 99.8% combined authentication accuracy

🔒 <0.005% fraud rate

👵 85%+ elderly usability success

# 🚀 Vision

Financial inclusion for 200M+ Indians left behind by UPI.
Your voice + face = your most secure digital wallet.