import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.data_loader import DAREDataset
from src.models.VideoMAETransformer import VideoMAEWrapper
from src.utils.input_loader import InputLoader


def main():
    # Load config
    input_loader = InputLoader()
    config = input_loader.load_config("configs/model_config_videomae.yaml")

    # Setup dataset
    dataset = DAREDataset(
        video_dir=config["video_dir"],
        annotation_file=config["annotation_file"],
        num_frames=config.get("num_frames", 16),
        frame_size=config.get("frame_size", 224),
        preload=config.get("preload", False),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoMAEWrapper.from_config(config).to(device)

    # Training loop (1 epoch for demo)
    model.train()
    for videos, labels in dataloader:
        videos, labels = videos.to(device), labels.to(device)
        logits = model(videos)  # (B, num_classes)
        loss = F.cross_entropy(logits, labels)

        print(f"Loss: {loss.item():.4f}")
        loss.backward()
        # optimizer.step(), etc.


if __name__ == "__main__":
    main()
