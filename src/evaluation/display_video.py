import os
import cv2

def display_video(video_path):
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    print(f"Playing {os.path.basename(video_path)}. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    subfolder = "test"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(base_dir, "deception_detection", subfolder)
    keyword = input("Enter video keyword (e.g. trial_truth_007): ").strip()
    video_files = [f for f in os.listdir(video_dir) if keyword in f and f.endswith(".mp4")]
    video_files.sort()  # Sort files by name
    if not video_files:
        print(f"No videos found with '{keyword}' in {video_dir}")
    else:
        for video_file in video_files:
            display_video(os.path.join(video_dir, video_file))