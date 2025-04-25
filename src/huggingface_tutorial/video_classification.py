import os

import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import pytorchvideo.data

import imageio
import numpy as np
from IPython.display import Image

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

from src.data.data_annotator import build_clip_label_map
from src.data.data_loader import DeceptionDataset
from src.utils import constants
import evaluate
from transformers import TrainingArguments, Trainer

video_clips_dir_train = os.path.join(constants.ROOT_DIR, "data", "train")
video_clips_dir_val = os.path.join(constants.ROOT_DIR, "data", "val")
video_clips_dir_test = os.path.join(constants.ROOT_DIR, "data", "test")

clip_to_label_train = build_clip_label_map(video_clips_dir_train)
clip_to_label_val = build_clip_label_map(video_clips_dir_val)
clip_to_label_test = build_clip_label_map(video_clips_dir_test)

class_labels = {"truth", "lie"}
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

model_ckpt = "MCG-NJU/videomae-base"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
# model = VideoMAEForVideoClassification.from_pretrained(
#     model_ckpt,
#     label2id=label2id,
#     id2label=id2label,
#     ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
# )

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = 16
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps

train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)


train_dataset = DeceptionDataset(
    video_label_map=clip_to_label_train,
    transform=train_transform
)
print(f"Loaded {len(train_dataset)} videos.")

val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

val_dataset = DeceptionDataset(
    video_label_map=clip_to_label_val,
    transform=val_transform
)

test_dataset = DeceptionDataset(
    video_label_map=clip_to_label_test,
    transform=val_transform
)

print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)


def unnormalize_img(img):
    """Un-normalizes the image pixels."""
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(video_tensor, filename="sample.gif"):
    """Prepares a GIF from a video tensor.
    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
        frames.append(frame_unnormalized)
    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename


def display_gif(video_tensor, gif_name="sample.gif"):
    """Prepares and displays a GIF from a video tensor."""
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)


sample_video = next(iter(train_dataset))
video_tensor = sample_video["video"]
display_gif(video_tensor)

model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned-deception-dataset"
num_epochs = 4
batch_size = 8

args = TrainingArguments(
    new_model_name,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# trainer = Trainer(
#     model,
#     args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     processing_class=image_processor,
#     compute_metrics=compute_metrics,
#     data_collator=collate_fn,
# )

# train_results = trainer.train()
