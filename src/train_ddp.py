"""
Distributed Data Parallel (DDP) Training Script for Trimodal Emotion Recognition
Uses PyTorch DDP for multi-GPU training
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from dataset import RAVDESSTrimodalDataset
from model import TrimodalClassifier
from utils import (
    save_checkpoint,
    evaluate_model,
    plot_confusion_matrix,
    plot_training_curves,
    print_classification_report,
    AverageMeter,
    get_lr,
    set_seed
)


def setup_ddp(rank, world_size):
    """Initialize DDP process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] DDP initialized")


def cleanup_ddp():
    """Cleanup DDP process group"""
    dist.destroy_process_group()


def create_distributed_dataloaders(
    metadata_csv,
    base_dir,
    batch_size,
    num_workers,
    rank,
    world_size,
    augment_train=True
):
    """Create distributed dataloaders"""
    
    # Create datasets
    train_dataset = RAVDESSTrimodalDataset(
        metadata_csv=metadata_csv,
        base_dir=base_dir,
        split='train',
        augment=augment_train
    )
    
    val_dataset = RAVDESSTrimodalDataset(
        metadata_csv=metadata_csv,
        base_dir=base_dir,
        split='val',
        augment=False
    )
    
    test_dataset = RAVDESSTrimodalDataset(
        metadata_csv=metadata_csv,
        base_dir=base_dir,
        split='test',
        augment=False
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_sampler


def train_one_epoch(model, train_loader, optimizer, scheduler, device, rank, epoch):
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch in pbar:
        # Move to device
        audio_input = batch['audio_input_values'].to(device, non_blocking=True)
        video_input = batch['video_pixel_values'].to(device, non_blocking=True)
        text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
        text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(
            audio_input_values=audio_input,
            video_pixel_values=video_input,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean().item()
        
        # Update meters
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(accuracy, labels.size(0))
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}',
                'lr': f'{get_lr(optimizer):.6f}'
            })
    
    return loss_meter.avg, acc_meter.avg


def validate(model, val_loader, device, rank):
    """Validate model"""
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            audio_input = batch['audio_input_values'].to(device, non_blocking=True)
            video_input = batch['video_pixel_values'].to(device, non_blocking=True)
            text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
            text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(
                audio_input_values=audio_input,
                video_pixel_values=video_input,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
            
            # Update meters
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(accuracy, labels.size(0))
            
            # Store predictions
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return loss_meter.avg, acc_meter.avg, np.array(all_predictions), np.array(all_labels)


def main(rank, world_size, args):
    """Main training function"""
    
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    
    # Only print from rank 0
    if rank == 0:
        print(f"\n{'='*60}")
        print("TRIMODAL EMOTION RECOGNITION - DDP TRAINING")
        print(f"{'='*60}")
        print(f"World Size: {world_size}")
        print(f"Batch Size per GPU: {args.batch_size}")
        print(f"Global Batch Size: {args.batch_size * world_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning Rate: {args.lr}")
        print(f"{'='*60}\n")
    
    # Set seed for reproducibility
    set_seed(args.seed + rank)
    
    # Create dataloaders
    if rank == 0:
        print("Creating dataloaders...")
    
    train_loader, val_loader, test_loader, train_sampler = create_distributed_dataloaders(
        metadata_csv=args.metadata_csv,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
        augment_train=True
    )
    
    # Create model
    if rank == 0:
        print("\nCreating model...")
    
    model = TrimodalClassifier(
        num_classes=8,
        freeze_audio_encoder=True,
        use_gradient_checkpointing=args.gradient_checkpointing
    )
    
    model = model.to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader),
        eta_min=1e-6
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, args.epochs + 1):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{args.epochs}")
            print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, rank, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, device, rank
        )
        
        # Gather metrics from all processes
        train_loss_tensor = torch.tensor(train_loss).to(device)
        train_acc_tensor = torch.tensor(train_acc).to(device)
        val_loss_tensor = torch.tensor(val_loss).to(device)
        val_acc_tensor = torch.tensor(val_acc).to(device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.AVG)
        
        train_loss = train_loss_tensor.item()
        train_acc = train_acc_tensor.item()
        val_loss = val_loss_tensor.item()
        val_acc = val_acc_tensor.item()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print metrics (only rank 0)
        if rank == 0:
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save checkpoint if best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = args.output_dir / f'best_model.pth'
                save_checkpoint(
                    model.module,  # Save unwrapped model
                    optimizer,
                    epoch,
                    val_loss,
                    val_acc,
                    checkpoint_path
                )
                print(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
    
    # Final evaluation on test set (only rank 0)
    if rank == 0:
        print(f"\n{'='*60}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'='*60}\n")
        
        test_loss, test_acc, test_preds, test_labels = validate(
            model, test_loader, device, rank
        )
        
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        
        # Plot results
        emotion_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Confusion matrix
        plot_confusion_matrix(
            test_labels,
            test_preds,
            emotion_names,
            save_path=args.output_dir / 'confusion_matrix.png',
            title='Test Set Confusion Matrix'
        )
        
        # Training curves
        plot_training_curves(
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            save_path=args.output_dir / 'training_curves.png'
        )
        
        # Classification report
        print_classification_report(
            test_labels,
            test_preds,
            emotion_names,
            save_path=args.output_dir / 'classification_report.txt'
        )
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*60}\n")
    
    # Cleanup
    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trimodal Emotion Recognition DDP Training')
    
    # Data paths
    parser.add_argument('--base_dir', type=str, default='/u/erdos/csga/as505/Multimodal-Emotion-Recognition',
                        help='Base directory of the project')
    parser.add_argument('--metadata_csv', type=str, default=None,
                        help='Path to metadata.csv')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    
    # Model options
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    
    args = parser.parse_args()
    
    # Setup paths
    args.base_dir = Path(args.base_dir)
    if args.metadata_csv is None:
        args.metadata_csv = args.base_dir / 'data' / 'metadata.csv'
    else:
        args.metadata_csv = Path(args.metadata_csv)
    
    args.output_dir = args.base_dir / args.output_dir
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get world size from environment (set by torchrun)
    world_size = int(os.environ.get('WORLD_SIZE', 2))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Launch training
    main(local_rank, world_size, args)
