import argparse
import models
import time
import torch
import utils
from datasets import create_split_dataloaders, evaluate_dataloader
import os

def train_model(model, train_dataloader, val_dataloader, num_epochs=10, device=None, start_epoch=0):
    """
    Train the model using train and validation dataloaders
    start_epoch: Epoch to start training from (for resuming)
    """
    
    device = utils.get_device()
    torch.manual_seed(123)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.1)
    
    model = model.to(device)
    
    print(f"Starting training from epoch {start_epoch + 1} to epoch {start_epoch + num_epochs}")
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (videos, labels) in enumerate(train_dataloader):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass timing
            forward_start = time.time()
            outputs = model(videos)
            forward_time = time.time() - forward_start
            
            # Loss computation timing
            loss_start = time.time()
            loss = criterion(outputs, labels)
            loss_time = time.time() - loss_start
            
            # Backward pass timing
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}")
                print(f"  Forward pass: {forward_time:.4f}s")
                print(f"  Loss computation: {loss_time:.4f}s")
                print(f"  Backward pass: {backward_time:.4f}s")
                print(f"  Total batch time: {forward_time + loss_time + backward_time:.4f}s")
                
                # Evaluate on training set
                train_loss, train_acc, train_eval_time = evaluate_dataloader(
                    model, train_dataloader, criterion, utils.get_device(), max_batches=5)
                
                # Evaluate on validation set
                val_loss, val_acc, val_eval_time = evaluate_dataloader(
                    model, val_dataloader, criterion, utils.get_device(), max_batches=5)
                
                print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% (eval: {train_eval_time:.2f}s)")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% (eval: {val_eval_time:.2f}s)")
                print("-" * 60)
        
        # Final evaluation for the epoch
        train_loss, train_acc, train_eval_time = evaluate_dataloader(
            model, train_dataloader, criterion, utils.get_device(), max_batches=100)
        val_loss, val_acc, val_eval_time = evaluate_dataloader(
            model, val_dataloader, criterion, utils.get_device(), max_batches=100)
        
        print(f"Epoch {epoch + 1} completed:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% (eval: {train_eval_time:.2f}s)")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% (eval: {val_eval_time:.2f}s)")
        print("=" * 70)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }

        # Create checkpoints directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        
        # Save with epoch number in filename
        checkpoint_path = f'checkpoints/point_classifier_epoch_{epoch + 1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Swin3D model for tennis point prediction')
    parser.add_argument('--data-dir', default='training_clips', 
                       help='Directory containing video clips with subfolders 0 and 1 corresponding to labels')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs to train (default: 5)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    parser.add_argument('--save-final', type=str, default='point_classifier.pth',
                       help='Path to save final model state (default: point_classifier.pth)')
    parser.add_argument('--debug', action='store_true',
                       help='Print debug information about data loaders')
    
    args = parser.parse_args()
    
    # Load model and prepare for fine-tuning
    print("Loading Swin3D model...")
    model, preprocess = models.load_video_swin_model()
    model = models.prepare_for_finetuning(model)
    
    print(f"Creating data loaders from {args.data_dir}...")
    train_loader, val_loader, test_loader = create_split_dataloaders(
        args.data_dir, transform=preprocess, batch_size=args.batch_size
    )
    
    if args.debug:
        print("\n=== Data Loader Debug Info ===")
        print(f"Train loader batches: {len(train_loader)}")
        print(f"Val loader batches: {len(val_loader)}")
        print(f"Test loader batches: {len(test_loader)}")
        
        # Get a sample batch from train loader
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            print(f"\nTrain batch {batch_idx}:")
            print(f"  Input shape: {inputs.shape}")
            print(f"  Label shape: {labels.shape}")
            print(f"  Label values: {labels}")
            print(f"  Input dtype: {inputs.dtype}")
            print(f"  Label dtype: {labels.dtype}")
            print(f"  Input min/max: {inputs.min():.4f} / {inputs.max():.4f}")
            break  # Only show first batch
        print("\n=== Data Loader Debug Complete ===")
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        start_epoch = models.load_checkpoint(args.checkpoint, model)
    
    print(f"\n=== Starting Training ===")
    train_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, 
        start_epoch=start_epoch
    )
    
    print(f"\nSaving final model to {args.save_final}")
    torch.save(model.state_dict(), args.save_final)
    print("Training complete")

if __name__ == "__main__":
    main()