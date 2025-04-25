import os
from typing import Dict


def build_clip_label_map(video_clips_dir: str) -> Dict[str, int]:
    """
    Maps each video clip path to its deception label (0 = truth, 1 = lie)
    based solely on its filename (e.g., trial_lie_002_002.mp4).

    Args:
        video_clips_dir (str): Path to directory containing segmented video clips.

    Returns:
        Dict[str, int]: Dictionary mapping each clip path to its numeric label.
    """
    clip_to_label = {}
    for fname in os.listdir(video_clips_dir):
        if not fname.endswith(".mp4"):
            continue

        parts = fname.split('_')
        if len(parts) < 3:
            print(f"Skipping malformed filename: {fname}")
            continue

        label_str = parts[1]
        label = 1 if label_str.lower() == "lie" else 0
        path = os.path.join(video_clips_dir, fname)
        clip_to_label[path] = label

    return clip_to_label
