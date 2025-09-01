import os
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from biometric import get_face_vector
from liveness import check_liveness
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

Base = declarative_base()
engine = create_engine("sqlite:///./ignivox_app.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    face_vector = Column(String, nullable=False)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Ignivox Face Integration V1")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/v1/face-enroll")
async def face_enroll(username: str, image: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = f"temp_{image.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await image.read())
    if not check_liveness(file_path):
        os.remove(file_path)
        raise HTTPException(status_code=403, detail="Face liveness check failed.")
    face_vector = get_face_vector(file_path)
    os.remove(file_path)
    if face_vector is None:
        raise HTTPException(status_code=400, detail="Face not detected or vector could not be extracted.")
    # Save user
    db_user = User(username=username, face_vector=",".join(map(str, face_vector)))
    db.add(db_user)
    db.commit()
    return {"status": "enrolled", "msg": "User enrolled successfully."}

@app.post("/api/v1/face-auth")
async def face_auth(username: str, image: UploadFile = File(...), db: Session = Depends(get_db)):
    file_path = f"temp_{image.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await image.read())
    if not check_liveness(file_path):
        os.remove(file_path)
        raise HTTPException(status_code=403, detail="Face liveness check failed.")
    face_vector = get_face_vector(file_path)
    os.remove(file_path)
    if face_vector is None:
        raise HTTPException(status_code=400, detail="Face not detected or vector could not be extracted.")
    # Get user & compare
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not registered.")
    db_vector = np.array([float(x) for x in user.face_vector.split(",")])
    sim = cosine_similarity([face_vector], [db_vector])[0][0]
    if sim < 0.6:  # You can tune this threshold
        raise HTTPException(status_code=401, detail="Face does not match.")
    return {
        "status": "success",
        "msg": "Face authentication successful.",
        "similarity": sim
    }
