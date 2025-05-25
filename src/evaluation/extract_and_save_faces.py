import os
import cv2
import face_recognition
from video_loader import load_video
from compare_faces import compare_faces_from_images

def group_faces_by_identity(face_paths):
    """
    Groups face image paths by identity using pairwise comparison.
    Returns a list of groups, each group is a list of image paths.
    """
    groups = []
    for face_path in face_paths:
        placed = False
        for group in groups:
            # Compare with the first face in the group
            if compare_faces_from_images(face_path, group[0]):
                group.append(face_path)
                placed = True
                break
        if not placed:
            groups.append([face_path])
    return groups

def save_face_images_from_video(video_name, subfolder="test", output_base_folder="extracted_faces", nth_frame=30):
    # This function saves a few frames of the person that is seen the most in one video
    base_name = os.path.splitext(video_name)[0]
    parts = base_name.split('_')
    if len(parts) >= 4:
        xyz = parts[2]
    else:
        xyz = "unknown"
    output_folder = os.path.join(output_base_folder, subfolder, xyz)
    os.makedirs(output_folder, exist_ok=True)

    cap = load_video(video_name, subfolder=subfolder)
    frame_count = 0
    face_count = 0
    saved_faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % nth_frame == 0:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)

            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_image = frame[top:bottom, left:right]
                face_filename = os.path.join(
                    output_folder, f"{base_name}_frame{frame_count}_face{i}.jpg"
                )
                cv2.imwrite(face_filename, face_image)
                saved_faces.append(face_filename)
                face_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {face_count} faces from {frame_count} frames.")

    # Group faces by identity
    print("Grouping extracted faces by identity...")
    groups = group_faces_by_identity(saved_faces)

    # Move faces to separate folders per identity (no duplicates)
    identity_folders = []
    for idx, group in enumerate(groups):
        identity_folder = os.path.join(output_folder, f"person_{idx+1}")
        os.makedirs(identity_folder, exist_ok=True)
        for face_path in group:
            new_path = os.path.join(identity_folder, os.path.basename(face_path))
            if not os.path.exists(new_path):
                os.rename(face_path, new_path)
        identity_folders.append(identity_folder)

    print(f"Grouped faces into {len(groups)} identities.")

    # Remove all original unsorted images from the output folder
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        if os.path.isfile(file_path) and file.startswith(base_name) and file.endswith(".jpg"):
            os.remove(file_path)

    # Keep only the folder with the most images if more than one person detected
    if len(identity_folders) > 1:
        max_folder = max(identity_folders, key=lambda f: len(os.listdir(f)))
        # Move all images from max_folder to output_folder (xyz)
        for file in os.listdir(max_folder):
            src = os.path.join(max_folder, file)
            dst = os.path.join(output_folder, file)
            if not os.path.exists(dst):
                os.rename(src, dst)
        # Remove all person folders
        for folder in identity_folders:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    os.remove(os.path.join(folder, file))
                os.rmdir(folder)
        print(f"Kept only the folder with the most images: {os.path.basename(max_folder)}")
    elif len(identity_folders) == 1:
        # Move images from the only person folder to output_folder (xyz)
        only_folder = identity_folders[0]
        for file in os.listdir(only_folder):
            src = os.path.join(only_folder, file)
            dst = os.path.join(output_folder, file)
            if not os.path.exists(dst):
                os.rename(src, dst)
        os.rmdir(only_folder)

if __name__ == "__main__":
    # Loop over all video files in the subfolders
    subfolder = "test"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_dir, "deception_detection", subfolder)
    for filename in os.listdir(video_dir):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print(f"\nProcessing {filename}...")
            save_face_images_from_video(filename, subfolder=subfolder)
    subfolder = "train"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_dir, "deception_detection", subfolder)
    for filename in os.listdir(video_dir):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print(f"\nProcessing {filename}...")
            save_face_images_from_video(filename, subfolder=subfolder)