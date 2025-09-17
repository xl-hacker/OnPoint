import os
import torch
import torchvision.models.video as models
from torchvision.models.video import Swin3D_B_Weights

def load_video_swin_model():
    weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
    model = models.swin3d_b(weights=weights)
    return model, weights.transforms()

def prepare_for_finetuning(model):
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # This is the final Sequential with 2 SwinTransformerBlocks    
    last_stage = model.features[6]
    
    # Unfreezing last_stage[-1] also works reasonably well instead
    # of unfreezing both.
    for param in last_stage[0].parameters():
        param.requires_grad = True

    for param in last_stage[1].parameters():
        param.requires_grad = True

    in_features = model.head.in_features
    num_classes = 2
    model.head = torch.nn.Linear(in_features=in_features, out_features=num_classes)
    return model

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load a checkpoint and return the epoch number and training state
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        model: The model to load state into
        optimizer: The optimizer to load state into (optional)
    
    Returns:
        int: The epoch number from the checkpoint, or 0 if loading failed
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")
        return 0
    
    try:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get epoch number
        start_epoch = checkpoint.get('epoch', 0)
        
        # Print checkpoint info
        print(f"Loaded checkpoint from epoch {start_epoch}")
        if 'train_loss' in checkpoint:
            print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        if 'train_acc' in checkpoint:
            print(f"  Train Acc: {checkpoint['train_acc']:.2f}%")
        if 'val_acc' in checkpoint:
            print(f"  Val Acc: {checkpoint['val_acc']:.2f}%")
        
        return start_epoch
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0

# Loads a fine-tuned model from saved weights
def load_trained_model(model_path, device):
    model, transforms = load_video_swin_model()
    model = prepare_for_finetuning(model)

    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model, transforms