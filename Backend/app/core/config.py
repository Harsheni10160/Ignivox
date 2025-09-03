from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    database_url: str = "postgresql://securepay_user:1234567890@localhost/securepay"
    redis_url: str = "redis://localhost:6379"
    secret_key: str = "DevaHarsheni"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Add Razorpay + Google credentials
    razorpay_key_id: Optional[str] = None
    razorpay_key_secret: Optional[str] = None
    google_application_credentials: Optional[str] = None
    
    class Config:
        env_file = ".env"

settings = Settings()