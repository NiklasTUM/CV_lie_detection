# Video-Based Deception Detection with Transformers

This repository contains all scripts and notebooks used in a computer vision project focused on detecting deception 
in video clips using transformer-based models.
It includes preprocessing utilities, training pipelines, qualitative evaluation tools, and model-specific configurations.

---

## Overview

The project is structured around two main approaches:

- **Visual-only deception detection using VideoMAE**, a self-supervised Vision Transformer optimized for video understanding.
- **Multimodal deception detection using InternVL**, a large-scale transformer capable of visual-text reasoning.

---

## Core Notebooks

### 1. `src/videomae/video_classification_mae.ipynb`
Implements deception detection using VideoMAE, fine-tuned on short 16-frame video chunks.

**Features:**
- Loads and preprocesses videos.
- Fine-tunes pre-trained VideoMAE models (e.g., Base, Large).
- Supports chunk- and scene-level prediction using:
  - Majority vote
  - Logit-sum (recommended)
  - Softmax-sum

---

### 2. `src/internvl/video_classification_internvl.ipynb`
Applies InternVL to full-scene classification as a text generation task.

**Features:**
- Processes entire video scenes instead of short clips.
- Uses autoregressive prompting for "truth" or "lie" token generation.
- Includes LoRA-based fine-tuning setup (optional).

**Notes:**
- Requires >40GB GPU (A100 recommended).
- Dataset format differs from VideoMAE.
- Some official scripts require manual correction.

---

## Data Processing Utilities

### `data/data_loader.py`
Wraps `datasets.Dataset` loading logic for transformer-based models.

### `data/data_annotator.py`
Generates deception labels (`0 = truth`, `1 = lie`) based on video filenames.

### `data/data_splitter.py`
Performs train/val/test split by scene ID, maintaining class balance.

---

## Face and Video Processing Scripts

### `video_splitter.py`
Splits long videos into uniform-length clips (default: 3–4 seconds).

### `fps_manipulator.py`
Adjusts the frame rate of all videos in a given folder.

### `display_video.py`, `video_loader.py`
Manually display single or multiple videos (useful for verification).

---

## Face Analysis Tools (Optional)

### `extract_and_save_faces.py`
Extracts and saves faces from each video; only the most frequent identity per video is kept.

### `compare_faces.py`
Compares two face images and returns similarity based on embedding distance.

### `face_counter.py`
Searches for overlapping identities between train/test splits using face encodings and copies matching results for review.

---

## Qualitative Evaluation

### `qualitative_evaluation.ipynb`
Performs inference on video clips using a pre-trained model and visualizes results.

**Outputs:**
- Chunk-level and scene-level predictions
- Confidence scores
- Summary plots

---

## Suggested Workflow

1. **Preprocess Videos**
   - Use `video_splitter.py` and `fps_manipulator.py` to prepare uniform, frame-consistent clips.

2. **Assign Labels and Split**
   - Use `data_annotator.py` to generate labels.
   - Use `data_splitter.py` to create reproducible and balanced splits.

3. **Model Training and Inference**
   - Run `video_classification_mae.ipynb` or `video_classification_internvl.ipynb` depending on desired model.
   - Apply appropriate aggregation strategy (e.g., logit-sum).

4. **Face Identity Checking (Optional)**
   - Use `extract_and_save_faces.py` and `face_counter.py` to check for identity leakage between splits.

5. **Qualitative Review**
   - Use `qualitative_evaluation.ipynb` to manually inspect model predictions and scene grouping.

