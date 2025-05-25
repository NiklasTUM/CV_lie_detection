import os
import shutil
import face_recognition
from multiprocessing import Pool, cpu_count

def get_person_folders(base_folder):
    return [os.path.join(base_folder, d) for d in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, d))]

def get_image_paths(person_folder):
    return [os.path.join(person_folder, f) for f in os.listdir(person_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def compute_encodings(image_paths):
    encodings = {}
    for img_path in image_paths:
        img = face_recognition.load_image_file(img_path)
        faces = face_recognition.face_encodings(img)
        if faces:
            encodings[img_path] = faces[0]
    return encodings

def compare_pair(args):
    test_img, test_enc, train_img, train_enc, threshold = args
    distance = face_recognition.face_distance([test_enc], train_enc)[0]
    return distance < threshold, test_img, train_img

def main():
    test_base = os.path.join("extracted_faces", "test")
    train_base = os.path.join("extracted_faces", "train")
    matches_base = os.path.join("extracted_faces", "matches")
    os.makedirs(matches_base, exist_ok=True)

    test_persons = get_person_folders(test_base)
    train_persons = get_person_folders(train_base)

    print("Precomputing encodings for test images...")
    test_encodings = {}
    for test_person in test_persons:
        test_images = get_image_paths(test_person)
        test_encodings[test_person] = compute_encodings(test_images)

    print("Precomputing encodings for train images...")
    train_encodings = {}
    for train_person in train_persons:
        train_images = get_image_paths(train_person)
        train_encodings[train_person] = compute_encodings(train_images)

    threshold = 0.5
    pool = Pool(processes=cpu_count())

    match_summary = {}

    for test_idx, test_person in enumerate(test_persons, 1):
        test_name = os.path.basename(test_person)
        test_imgs_enc = test_encodings[test_person]
        match_count = 0
        print(f"\nComparing test person {test_idx}/{len(test_persons)}: {test_name}")
        for train_idx, train_person in enumerate(train_persons, 1):
            train_name = os.path.basename(train_person)
            train_imgs_enc = train_encodings[train_person]
            print(f"  Against train person {train_idx}/{len(train_persons)}: {train_name}")

            # Prepare all pairs for multiprocessing
            tasks = [
                (test_img, test_enc, train_img, train_enc, threshold)
                for test_img, test_enc in test_imgs_enc.items()
                for train_img, train_enc in train_imgs_enc.items()
            ]

            results = pool.map(compare_pair, tasks)
            found_match = False
            for is_match, test_img, train_img in results:
                if is_match:
                    match_count += 1
                    found_match = True
                    match_folder = os.path.join(matches_base, f"match_{test_name}_{train_name}")
                    os.makedirs(match_folder, exist_ok=True)
                    shutil.copy2(test_img, os.path.join(match_folder, f"test_{os.path.basename(test_img)}"))
                    shutil.copy2(train_img, os.path.join(match_folder, f"train_{os.path.basename(train_img)}"))
                    break  # Only need one match per person pair
            if found_match:
                continue
        match_summary[test_name] = match_count
        print(f"{test_name} appears {match_count} times in the train folder.")

    pool.close()
    pool.join()

    print("\n=== Match Summary ===")
    for test_name, match_count in match_summary.items():
        print(f"{test_name}: {match_count} matches in the train folder.")

if __name__ == "__main__":
    main()