from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.schemas.user import UserCreate, UserResponse
from app.services.user_service import create_user, authenticate_user

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Implementation here
    pass

@router.post("/login")
async def login_for_access_token(phone: str, db: Session = Depends(get_db)):
    # Implementation here
    pass