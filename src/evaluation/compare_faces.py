import face_recognition
from face_recognition import face_distance

def compare_faces_from_images(image_1_path, image_2_path):
    image_1 = face_recognition.load_image_file(image_1_path)
    image_2 = face_recognition.load_image_file(image_2_path)

    encodings_1 = face_recognition.face_encodings(image_1)
    encodings_2 = face_recognition.face_encodings(image_2)

    if not encodings_1 or not encodings_2:
        # If either image has no face, treat as not matching
        return False

    encoding_1 = encodings_1[0]
    encoding_2 = encodings_2[0]
    
    distance = face_distance([encoding_1], encoding_2)[0]

    THRESHOLD = 0.50 # 0.5 for strict, 0.6 more loose
    return distance < THRESHOLD