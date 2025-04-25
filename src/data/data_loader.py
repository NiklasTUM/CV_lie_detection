import torch
from pytorchvideo.transforms import UniformTemporalSubsample
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from typing import Callable, Dict


class DeceptionDataset(Dataset):
    def __init__(
        self,
        video_label_map: Dict[str, int],
        transform: Callable = None,
        num_frames: int = 16,
    ):
        self.video_label_map = video_label_map
        self.video_paths = list(video_label_map.keys())
        self.transform = transform
        self.num_frames = num_frames
        self.num_videos = len(video_label_map)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.video_label_map[video_path]

        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(0, video.duration)
        video_tensor = video_data['video']  # shape: (C, T, H, W)

        if video_tensor is None:
            raise ValueError(f"Could not load video: {video_path}")

        C, T, H, W = video_tensor.shape
        if T < self.num_frames:
            pad = video_tensor[:, -1:].repeat(1, self.num_frames - T, 1, 1)
            video_tensor = torch.cat([video_tensor, pad], dim=1)

        # Subsample T frames uniformly
        video_clip = UniformTemporalSubsample(self.num_frames)(video_tensor)  # still (C, T, H, W)

        if self.transform:
            video_clip = self.transform({"video": video_clip})["video"]

        return {
            "video": video_clip,  # torch.Tensor of shape (C, T, H, W)
            "label": label        # 0 (truth) or 1 (lie)
        }
