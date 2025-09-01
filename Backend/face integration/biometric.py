from deepface import DeepFace

def get_face_vector(image_path):
    embedding_objs = DeepFace.represent(img_path=image_path, model_name="Facenet")
    if embedding_objs:
        return embedding_objs[0]["embedding"]
    return None


