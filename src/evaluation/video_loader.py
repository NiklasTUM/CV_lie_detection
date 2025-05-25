import os
import cv2

def load_video(video_name, subfolder="test"):
    """
    Loads a video from the specified subfolder inside the 'files' directory.

    Args:
        video_name (str): Name of the video file (e.g., 'sample.mp4').
        subfolder (str): Subfolder inside 'files' where the video is stored.

    Returns:
        cv2.VideoCapture: OpenCV VideoCapture object.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "deception_detection", subfolder, video_name)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    return cap


# Example usage:
if __name__ == "__main__":
    cap = load_video("trial_lie_002_000.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()