import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Optional


class DAREDataset(Dataset):
    def __init__(
        self,
        video_dir: str,
        annotation_file: str,
        num_frames: int = 16,
        transform: Optional[transforms.Compose] = None,
        frame_size: int = 224,
        preload: bool = False
    ):
        self.video_dir = video_dir
        self.annotation_file = annotation_file
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor()
        ])
        self.samples = self._load_annotations()
        self.preload = preload
        if preload:
            self.preloaded_data = [self._load_video(path) for path, _ in self.samples]

    def _load_annotations(self) -> List:
        with open(self.annotation_file, 'r') as f:
            lines = f.readlines()
        samples = []
        for line in lines:
            name, label = line.strip().split()
            video_path = os.path.join(self.video_dir, f"{name}.avi")
            samples.append((video_path, int(label)))
        return samples

    def _load_video(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        frames = []
        for idx in range(total_frames):
            ret, frame = cap.read()
            if idx in frame_indices and ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(frame))
        cap.release()
        if len(frames) < self.num_frames:
            # Pad with last frame
            last_frame = frames[-1] if frames else torch.zeros(3, self.frame_size, self.frame_size)
            while len(frames) < self.num_frames:
                frames.append(last_frame)
        return torch.stack(frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            video = self.preloaded_data[idx]
        else:
            video = self._load_video(self.samples[idx][0])
        label = self.samples[idx][1]
        return video, label
