import os
import shutil
import random
from collections import defaultdict
from typing import Tuple, List

from src.utils import constants


def extract_clip_info(filename: str) -> Tuple[str, str]:
    # Parses 'trial_lie_001_000.mp4' → ('lie', '001')
    parts = filename.split('_')
    label = parts[1]
    scene_id = parts[2]
    return label, scene_id

def split_deception_clips(video_dir: str, output_dir: str, seed: int = 42):
    random.seed(seed)

    # Group clips by (label, scene_id)
    grouped_clips = defaultdict(list)

    for fname in os.listdir(video_dir):
        if not fname.endswith(".mp4"):
            continue
        label, scene_id = extract_clip_info(fname)
        key = (label, scene_id)
        grouped_clips[key].append(fname)

    # Separate by class
    scene_ids_by_label = defaultdict(set)
    for (label, scene_id), clips in grouped_clips.items():
        scene_ids_by_label[label].add(scene_id)

    # Find minimum available scenes per class
    min_scenes = min(len(scene_ids_by_label['lie']), len(scene_ids_by_label['truth']))
    num_train = int(min_scenes * 0.8)
    num_val = int(min_scenes * 0.1)
    num_test = min_scenes - num_train - num_val

    # Create splits
    splits = {'train': set(), 'val': set(), 'test': set()}
    for label in ['lie', 'truth']:
        scenes = list(scene_ids_by_label[label])
        random.shuffle(scenes)
        splits['train'].update((label, sid) for sid in scenes[:num_train])
        splits['val'].update((label, sid) for sid in scenes[num_train:num_train+num_val])
        splits['test'].update((label, sid) for sid in scenes[num_train+num_val:num_train+num_val+num_test])

    # Prepare output folders
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Copy clips
    for (label, scene_id), filenames in grouped_clips.items():
        for split, split_keys in splits.items():
            if (label, scene_id) in split_keys:
                for fname in filenames:
                    src = os.path.join(video_dir, fname)
                    dst = os.path.join(output_dir, split, fname)
                    shutil.copyfile(src, dst)
                break  # prevent copying into multiple splits

    print("Split complete. Videos saved to:", output_dir)


if __name__ == "__main__":
    video_clips_dir = os.path.join(constants.ROOT_DIR, "data", "Video_chunks")
    video_clips_output = os.path.join(constants.ROOT_DIR, "data")

    split_deception_clips(video_clips_dir, video_clips_output)
