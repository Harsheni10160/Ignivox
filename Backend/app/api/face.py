from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.models.user import User
import face_recognition
import numpy as np

router = APIRouter(prefix="/face", tags=["face-authentication"])

def _file_to_encoding(image_bytes: bytes) -> np.ndarray | None:
    try:
        # Load image from bytes
        import io
        from PIL import Image

        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            return encodings[0]
        return None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

@router.post("/enroll")
async def enroll_face(user_id: int, image: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await image.read()
    encoding = _file_to_encoding(content)
    if encoding is None:
        raise HTTPException(status_code=400, detail="No face detected in image")

    user: User | None = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Store as JSON string
    user.face_template = ",".join(map(str, encoding.tolist()))
    db.add(user)
    db.commit()
    return {"success": True, "message": "Face enrolled"}

@router.post("/verify")
async def verify_face(user_id: int, image: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await image.read()
    probe = _file_to_encoding(content)
    if probe is None:
        return {"verified": False, "message": "No face detected"}

    user: User | None = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.face_template:
        return {"verified": False, "message": "No enrolled face"}

    try:
        stored = np.array(list(map(float, user.face_template.split(","))))
    except Exception:
        raise HTTPException(status_code=500, detail="Corrupt stored face template")

    result = face_recognition.compare_faces([stored], probe)
    return {"verified": bool(result[0])}


