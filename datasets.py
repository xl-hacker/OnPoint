import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from pathlib import Path
import time
import utils


class VideoClipDataset(Dataset):
    def __init__(self, clips, labels, num_frames=16, transform=None):
        self.num_frames = num_frames
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = utils.get_swin3d_transform()

        self.clips = clips
        self.labels = labels
        print(f"Loaded {len(self.clips)} clips with labels: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        video_path = self.clips[idx]
        label = self.labels[idx]
        
        frames = utils.extract_frames_uniform(video_path, self.num_frames)
        video_tensor = utils.preprocess_frames_swin(frames, self.transform)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)


def _split_dataset(clips_folder, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Returns:
        tuple: (train_clips, val_clips, test_clips, train_labels, val_labels, test_labels)
    """
    from sklearn.model_selection import train_test_split
    import random
    
    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)
    
    clips = []
    labels = []
    
    clips_path = Path(clips_folder)
    for label in [0, 1]:
        label_folder = clips_path / str(label)
        if label_folder.exists():
            for clip_file in label_folder.glob('*.mp4'):
                clips.append(str(clip_file))
                labels.append(label)
    
    if len(clips) == 0:
        raise ValueError(f"No clips found in {clips_folder}")
    
    clips = np.array(clips)
    labels = np.array(labels)
    
    print(f"Total clips: {len(clips)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    train_clips, temp_clips, train_labels, temp_labels = train_test_split(
        clips, labels, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state,
        stratify=labels
    )
    
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_clips, test_clips, val_labels, test_labels = train_test_split(
        temp_clips, temp_labels,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_state,
        stratify=temp_labels
    )
    
    print(f"Training set: {len(train_clips)} clips")
    print(f"Validation set: {len(val_clips)} clips")
    print(f"Test set: {len(test_clips)} clips")
    print(f"Training labels: {np.bincount(train_labels)}")
    print(f"Validation labels: {np.bincount(val_labels)}")
    print(f"Test labels: {np.bincount(test_labels)}")
    
    return train_clips, val_clips, test_clips, train_labels, val_labels, test_labels


def _create_split_datasets(clips_folder, transform=None, val_ratio=0.15, test_ratio=0.15,
                         num_frames=16, random_state=42):
    """
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Split the dataset
    train_clips, val_clips, test_clips, train_labels, val_labels, test_labels = _split_dataset(
        clips_folder, val_ratio, test_ratio, random_state
    )

    # Create datasets
    train_dataset = VideoClipDataset(
        clips=train_clips,
        labels=train_labels,
        transform=transform,
        num_frames=num_frames,
    )
    
    val_dataset = VideoClipDataset(
        clips=val_clips,
        labels=val_labels,
        transform=transform,
        num_frames=num_frames,
    )
    
    test_dataset = VideoClipDataset(
        clips=test_clips,
        labels=test_labels,
        transform=transform,
        num_frames=num_frames,
    )
    
    return train_dataset, val_dataset, test_dataset


def create_split_dataloaders(clips_folder, transform=None, batch_size=4,
                             val_ratio=0.15, test_ratio=0.15, num_frames=16,
                             num_workers=0, shuffle_train=True, random_state=42):
    """
    clips_folder needs to contain two subfolders with names 0 and 1 (respective labels)
    that have individual clips as .mp4 files (3s snippets work well).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset, val_dataset, test_dataset = _create_split_datasets(
        clips_folder, transform, val_ratio, test_ratio, num_frames, random_state
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

def evaluate_dataloader(model, dataloader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    
    eval_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy, eval_time