# OnPoint - Tennis Video Classification

Fine-tuning video Swin transformer for detection of tennis points/rallies using PyTorch.

## Overview

This project uses a pre-trained Swin3D model to classify tennis video clips into two categories:
- **Class 0**: Non-point/rally clips
- **Class 1**: Point/rally clips

## Features

- Video preprocessing with uniform frame extraction
- Swin3D model fine-tuning with frozen backbone
- Training with validation and checkpointing
- Support for resuming training from checkpoints

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- scikit-learn
- PIL/Pillow
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xl-hacker/OnPoint.git
cd OnPoint
```

2. Create and activate a virtual environment:
```bash
python3 -m venv onpoint
source onpoint/bin/activate  # On Windows: onpoint\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

Organize your video clips in the following structure (sample provided):
```
training_clips/
├── 0/          # Non-point/rally clips
│   ├── clip1.mp4
│   ├── clip2.mp4
│   └── ...
└── 1/          # Point/rally clips
    ├── clip1.mp4
    ├── clip2.mp4
    └── ...
```

## Creating Training Clips

If you have longer tennis videos and want to create training clips with proper labels, use the `create_snippets.py` script:

### Basic Usage

Create 3-second clips from a video with timestamp-based labeling:
```bash
python create_snippets.py video.mp4 --timestamps timestamps.txt
```

### Timestamp File Format

Create a text file with timestamp ranges where points/rallies occur (one per line):
```
[45.2, 67.8]
[120.5, 145.3]
[200.1, 225.7]
```
## Usage

### Training

Basic training with default parameters:
```bash
python train.py --data-dir training_clips
```

### Resume Training

Resume training from a checkpoint:
```bash
python train.py \
    --data-dir training_clips \
    --checkpoint checkpoints/point_classifier_epoch_5.pth \
    --epochs 10
```

## Model Architecture

- **Base Model**: Swin3D-B pre-trained on Kinetics-400
- **Fine-tuning Strategy**: Freeze backbone, unfreeze last transformer blocks
- **Output**: 2-class classification (non-point vs point/rally)
- **Input**: 16 frames per video clip, resized to 224x224

## Project Structure

```
OnPoint/
├── train.py           # Main training script
├── models.py          # Model loading and fine-tuning functions
├── datasets.py        # Dataset classes and data loading utilities
├── utils.py           # Video processing and utility functions
├── create_snippets.py # Script to create training clips from longer videos
├── interactive.py     # Interactive inference (placeholder)
├── requirements.txt   # Python dependencies
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Swin3D model from Microsoft Research
- PyTorch and torchvision for deep learning framework
- OpenCV for video processing
