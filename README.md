# Video-Based Deception Detection with Transformers

This repository contains two Jupyter notebooks for experimenting with video-based lie detection using transformer-based architectures. The models are applied to short video clips and are evaluated based on chunk-level and scene-level predictions.

---

## Notebooks

### 1. `src/videomae/video_classification_mae.ipynb`
This notebook implements deception detection using [VideoMAE](https://arxiv.org/abs/2203.12602), a self-supervised Vision Transformer (ViT) architecture tailored for video understanding.

#### 💡 Features:
- Loads and preprocesses short video clips (e.g., 16-frame segments).
- Fine-tunes pre-trained VideoMAE models (Base, Large).
- Performs inference and scene-level prediction via:
  - Majority vote
  - Logit-sum voting
  - Softmax-sum voting

#### 🔧 Configuration:
- Specify model name in `model_name_or_path`, e.g., `MCG-NJU/VideoMAE-Base`.
- Video frames must be preprocessed to match expected input size (e.g., 224×224).
- Uses Hugging Face Transformers and Datasets.

---

### 2. `src/internvl/video_classification_internvl.ipynb`
This notebook attempts to apply the [InternVL](https://github.com/OpenGVLab/InternVL) multimodal transformer model for full-scene deception classification.

#### 💡 Features:
- Loads full video scenes as input.
- Reformulates classification as a generation task: the model predicts `"truth"` or `"lie"` tokens.
- Supports optional LoRA-based fine-tuning.

#### ⚠️ Notes:
- InternVL requires significant GPU memory (>40GB recommended).
- The official training script may need manual fixes.
- The dataset format required for LoRA fine-tuning differs from the format required by VideoMAE.

## 📁 data

The data directory contains three python scripts. `data_annotator.py` contains the function to build the mapping of 
each video clip path to its deception label (0 = truth, 1 = lie) based solely on its filename (e.g., trial_lie_002_002.mp4).
`data_loader.py` contains the "DeceptionDetection" wrapper class needed for the correct loading of videos for the 
fine-tuning of the VideoMAE Transformer. `data_splitter.py` contains the function to split the dataset into train, validation
and test split while ensuring all splits are balanced.

