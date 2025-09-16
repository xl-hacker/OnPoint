import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Resize, CenterCrop, Normalize
from torchvision.transforms import Compose

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    return device

def extract_frames_uniform(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= num_frames:
        # If video has fewer frames than needed, repeat frames
        frame_indices = list(range(total_frames))
        while len(frame_indices) < num_frames:
            frame_indices.extend(frame_indices[:num_frames - len(frame_indices)])
    else:
        # Extract frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        else:
            # If frame read fails, use a black frame
            black_frame = Image.new('RGB', (224, 224), (0, 0, 0))
            frames.append(black_frame)
    
    cap.release()
    return frames

def get_swin3d_transform():
    return Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_frames_swin(frames, preprocess):
    """
    Prepare PIL Images for Swin3D input
    First convert PIL Images to tensors, then apply Swin3D preprocessing
    """
    # Basic preprocessing for individual frames (PIL -> tensor)
    basic_preprocess = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert PIL Images to tensors
    processed_frames = []
    for frame in frames:
        frame_tensor = basic_preprocess(frame)  # Shape: (C, H, W)
        processed_frames.append(frame_tensor)
    
    # Stack frames as (T, C, H, W)
    temporal_stack = torch.stack(processed_frames)  # Shape: (T, C, H, W)
    
    # Apply Swin3D preprocessing to the temporal stack
    # The preprocess function expects (T, C, H, W) and outputs (C, T, H, W)
    return preprocess(temporal_stack)  # Shape: (C, T, H, W)

# For 3D models, you need (batch, channels, time, height, width)
def prepare_3d_input(frames, preprocess):
    # Add batch dimension: (1, C, T, H, W)
    return preprocess_frames_swin(frames, preprocess).unsqueeze(0)

def encode_video(video_path, preprocess):
    # Load and process real video
    frames = extract_frames_uniform(video_path, num_frames=16)
    return prepare_3d_input(frames, preprocess)