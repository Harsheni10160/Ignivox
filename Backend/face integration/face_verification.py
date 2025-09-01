import face_recognition

def verify_face(new_image_path, stored_encoding):
    image = face_recognition.load_image_file(new_image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        result = face_recognition.compare_faces([stored_encoding], encodings[0])
        return result[0]  # True/False
    return False
