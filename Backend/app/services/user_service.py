from app.schemas.user import UserCreate, UserResponse
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dummy user DB for testing
fake_users_db = {}

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_user(user: UserCreate) -> UserResponse:
    hashed_pw = get_password_hash(user.password)
    new_user = {
        "id": len(fake_users_db) + 1,
        "username": user.username,
        "email": user.email,
        "password": hashed_pw,
    }
    fake_users_db[user.email] = new_user
    return UserResponse(**new_user)

def authenticate_user(email: str, password: str):
    user = fake_users_db.get(email)
    if not user or not verify_password(password, user["password"]):
        return None
    return UserResponse(**user)
