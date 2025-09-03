import speech_recognition as sr
import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class PaymentIntent:
    action: str  # 'send_money', 'check_balance', etc.
    amount: Optional[float] = None
    recipient: Optional[str] = None
    currency: str = 'INR'
    confidence: float = 0.0

class SpeechToTextService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        
        # Language support for Indian languages
        self.supported_languages = {
            'english': 'en-IN',
            'hindi': 'hi-IN',
            'tamil': 'ta-IN',
            'telugu': 'te-IN',
            'kannada': 'kn-IN',
            'bengali': 'bn-IN'
        }
    
    def convert_audio_to_text(self, audio_data, language='en-IN') -> Tuple[str, float]:
        """
        Convert audio to text using speech recognition
        Returns (text, confidence)
        """
        try:
            # Convert numpy array to bytes for AudioData
            if isinstance(audio_data, np.ndarray):
                # Ensure audio is in the right format
                audio_data = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_data.tobytes()
            else:
                audio_bytes = audio_data
            
            # Create audio source from audio data
            audio_source = sr.AudioData(
                audio_bytes, 
                sample_rate=16000, 
                sample_width=2
            )
            
            # Method 1: Try Google Speech Recognition
            try:
                print("Trying Google Speech Recognition...")
                text = self.recognizer.recognize_google(
                    audio_source, 
                    language=language,
                    show_all=False
                )
                print(f"Google Speech Recognition result: '{text}'")
                return text.lower(), 0.9
                
            except sr.RequestError as e:
                print(f"Google Speech Recognition failed: {e}")
                # Fallback to offline recognition
                try:
                    print("Trying Sphinx offline recognition...")
                    text = self.recognizer.recognize_sphinx(audio_source)
                    print(f"Sphinx recognition result: '{text}'")
                    return text.lower(), 0.7
                except Exception as e2:
                    print(f"Sphinx recognition failed: {e2}")
                    
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                # Try with different language settings
                try:
                    print("Trying with English-US...")
                    text = self.recognizer.recognize_google(
                        audio_source, 
                        language='en-US',
                        show_all=False
                    )
                    print(f"English-US recognition result: '{text}'")
                    return text.lower(), 0.8
                except:
                    pass
                    
                # Try with Hindi
                try:
                    print("Trying with Hindi...")
                    text = self.recognizer.recognize_google(
                        audio_source, 
                        language='hi-IN',
                        show_all=False
                    )
                    print(f"Hindi recognition result: '{text}'")
                    return text.lower(), 0.8
                except:
                    pass
                    
                return "", 0.0
                    
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return "", 0.0
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on common words
        """
        hindi_words = ['रुपये', 'पैसे', 'भेजो', 'दो', 'को']
        tamil_words = ['ரூபாய்', 'அனுப்பு', 'கொடு']
        
        if any(word in text for word in hindi_words):
            return 'hi-IN'
        elif any(word in text for word in tamil_words):
            return 'ta-IN'
        else:
            return 'en-IN'

class IntentRecognitionService:
    def __init__(self):
        # Payment-related keywords and patterns
        self.payment_patterns = {
            'send_money': [
                r'send (?P<amount>\d+) (?:rupees?|rs\.?) to (?P<recipient>[\w\s]+)',
                r'pay (?P<amount>\d+) to (?P<recipient>[\w\s]+)',
                r'transfer (?P<amount>\d+) to (?P<recipient>[\w\s]+)',
                r'give (?P<amount>\d+) (?:rupees?|rs\.?) to (?P<recipient>[\w\s]+)',
                # Hindi patterns
                r'(?P<recipient>[\w\s]+) को (?P<amount>\d+) रुपये भेजो',
                r'(?P<recipient>[\w\s]+) को (?P<amount>\d+) दो',
                # Simple patterns
                r'send (?P<amount>\d+) to (?P<recipient>[\w\s]+)',
                r'(?P<amount>\d+) to (?P<recipient>[\w\s]+)',
                r'(?P<recipient>[\w\s]+) (?P<amount>\d+)',
            ],
            'check_balance': [
                r'(?:what.s my|check my|show my) balance',
                r'balance check',
                r'how much money',
                r'मेरा बैलेंस क्या है',
                r'बैलेंस दिखाओ'
            ]
        }
        
        # Common recipient name variations
        self.name_variations = {
            'mom': ['mom', 'mummy', 'mother', 'ma', 'mama'],
            'dad': ['dad', 'daddy', 'father', 'papa', 'baba'],
            'wife': ['wife', 'biwi', 'patni'],
            'husband': ['husband', 'pati'],
            'son': ['son', 'beta', 'ladka'],
            'daughter': ['daughter', 'beti', 'ladki']
        }
    
    def extract_payment_intent(self, text: str) -> PaymentIntent:
        """
        Extract payment intent from recognized text
        """
        text = text.strip().lower()
        
        # Try to match payment patterns
        for action, patterns in self.payment_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if action == 'send_money':
                        return self._extract_send_money_intent(match, text)
                    elif action == 'check_balance':
                        return PaymentIntent(
                            action='check_balance',
                            confidence=0.9
                        )
        
        # Fallback: try to extract numbers and names
        return self._extract_fallback_intent(text)
    
    def _extract_send_money_intent(self, match: re.Match, original_text: str) -> PaymentIntent:
        """
        Extract send money intent from regex match
        """
        try:
            amount = float(match.group('amount'))
            recipient = match.group('recipient').strip()
            
            # Normalize recipient name
            recipient = self._normalize_recipient_name(recipient)
            
            return PaymentIntent(
                action='send_money',
                amount=amount,
                recipient=recipient,
                confidence=0.95
            )
        except (AttributeError, ValueError):
            return PaymentIntent(action='unknown', confidence=0.0)
    
    def _extract_fallback_intent(self, text: str) -> PaymentIntent:
        """
        Fallback intent extraction using simple patterns
        """
        # Extract numbers (likely amounts)
        numbers = re.findall(r'\d+', text)
        
        # Extract potential names (words that are not common words)
        common_words = {
            'send', 'pay', 'give', 'transfer', 'to', 'rupees', 'rs', 'money',
            'भेजो', 'दो', 'को', 'रुपये', 'पैसे'
        }
        
        words = text.split()
        potential_names = [
            word for word in words 
            if word not in common_words and not word.isdigit()
        ]
        
        if numbers and potential_names:
            return PaymentIntent(
                action='send_money',
                amount=float(numbers[0]),
                recipient=' '.join(potential_names[:2]),  # Take first 2 words as name
                confidence=0.7
            )
        
        return PaymentIntent(action='unknown', confidence=0.0)
    
    def _normalize_recipient_name(self, name: str) -> str:
        """
        Normalize recipient names (handle variations)
        """
        name_lower = name.lower().strip()
        
        for standard_name, variations in self.name_variations.items():
            if name_lower in variations:
                return standard_name
                
        return name.title()  # Return with proper capitalization
    
    def get_confirmation_text(self, intent: PaymentIntent) -> str:
        """
        Generate confirmation text for the intent
        """
        if intent.action == 'send_money':
            return f"Sending ₹{intent.amount} to {intent.recipient}. Please confirm."
        elif intent.action == 'check_balance':
            return "Checking your account balance."
        else:
            return "I didn't understand that command. Please try again."

# Main service that combines everything
class VoicePaymentService:
    def __init__(self):
        self.speech_service = SpeechToTextService()
        self.intent_service = IntentRecognitionService()
    
    def process_voice_command(self, audio_data) -> Tuple[PaymentIntent, str]:
        """
        Process complete voice command: Speech-to-Text -> Intent Recognition
        """
        print("=== Voice Command Processing ===")
        
        # Convert speech to text
        text, speech_confidence = self.speech_service.convert_audio_to_text(audio_data)
        
        print(f"Recognized text: '{text}'")
        print(f"Speech confidence: {speech_confidence}")
        
        if not text:
            print("No text recognized from audio")
            return PaymentIntent(action='error', confidence=0.0), "Could not understand audio"
        
        # Extract intent
        intent = self.intent_service.extract_payment_intent(text)
        print(f"Extracted intent: {intent}")
        
        # Adjust confidence based on speech recognition confidence
        intent.confidence *= speech_confidence
        
        # Generate confirmation text
        confirmation = self.intent_service.get_confirmation_text(intent)
        print(f"Confirmation text: {confirmation}")
        
        print("=== End Voice Command Processing ===")
        return intent, confirmation