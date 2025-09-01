import cv2
import mediapipe as mp

def check_liveness(image_path):
    mp_face = mp.solutions.face_mesh
    image = cv2.imread(image_path)
    if image is None:
        # Could not read the image
        return False
    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            return True
        return False


