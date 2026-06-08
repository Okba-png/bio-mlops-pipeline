"""
Training pipeline for Protein Secondary Structure Prediction.

This script trains a Transformer model on the protein dataset and evaluates
its performance. It saves checkpoints and generates plots of the training process.
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import ProteinDataset, collate_fn, AA_ALPHABET
from src.model import ProteinTransformer


def set_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, int]:
    """
    Calculates accuracy only for active (non-padded) elements.
    
    Args:
        logits: Shape (batch_size, seq_len, num_classes)
        targets: Shape (batch_size, seq_len) with -100 for padding.
        
    Returns:
        accuracy: Fraction of correct predictions.
        total_active: Number of valid residues evaluated.
    """
    predictions = logits.argmax(dim=-1)  # (batch_size, seq_len)
    
    # Mask out padding (-100)
    active_mask = (targets != -100)
    
    correct = (predictions == targets) & active_mask
    
    total_correct = correct.sum().item()
    total_active = active_mask.sum().item()
    
    accuracy = (total_correct / total_active) if total_active > 0 else 0.0
    return accuracy, total_active


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """Runs one training epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_residues = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for seqs, targets, masks in progress_bar:
        seqs = seqs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(seqs, src_key_padding_mask=masks)
        
        # Flatten tensors for loss computation:
        # logits: (batch_size * seq_len, num_classes)
        # targets: (batch_size * seq_len)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * seqs.size(0)
        acc, count = calculate_accuracy(logits, targets)
        total_correct += int(acc * count)
        total_residues += count
        
        progress_bar.set_postfix(loss=loss.item(), acc=acc)
        
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = (total_correct / total_residues) if total_residues > 0 else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """Evaluates the model on the validation or test set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_residues = 0
    
    for seqs, targets, masks in dataloader:
        seqs = seqs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        
        logits = model(seqs, src_key_padding_mask=masks)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        total_loss += loss.item() * seqs.size(0)
        acc, count = calculate_accuracy(logits, targets)
        total_correct += int(acc * count)
        total_residues += count
        
    eval_loss = total_loss / len(dataloader.dataset)
    eval_acc = (total_correct / total_residues) if total_residues > 0 else 0.0
    return eval_loss, eval_acc


def plot_learning_curves(
    history: dict[str, list[float]], 
    save_path: str
) -> None:
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Accuracy Curves (Residue level)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer on Protein Secondary Structure Dataset.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to val.csv")
    parser.add_argument("--test_csv", type=str, default=None, help="Path to test.csv (optional)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer embedding dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of encoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="Feedforward network dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--num_classes", type=int, default=3, choices=[3, 8], help="Number of structure classes")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model weights")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of dataset to use for training (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    print("Loading datasets...")
    train_dataset = ProteinDataset(args.train_csv, num_classes=args.num_classes)
    val_dataset = ProteinDataset(args.val_csv, num_classes=args.num_classes)
    
    # Apply subset fraction if specified
    if args.subset_fraction < 1.0:
        # Train subset
        train_size = int(len(train_dataset) * args.subset_fraction)
        train_indices = random.sample(range(len(train_dataset)), max(1, train_size))
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        # Val subset
        val_size = int(len(val_dataset) * args.subset_fraction)
        val_indices = random.sample(range(len(val_dataset)), max(1, val_size))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"Using subset fraction: {args.subset_fraction}. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Initialize Model
    model = ProteinTransformer(
        vocab_size=len(AA_ALPHABET),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)
    
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Loss and Optimizer
    # -100 is the default ignore_index and matches our padding value in collate_fn
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.save_dir, "best_model.pt")
    
    # Training Loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%"
        )
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': args
            }, best_model_path)
            print(f"  --> Saved new best model checkpoint to {best_model_path}")
            
    # Save training curves
    plot_path = os.path.join(args.save_dir, "learning_curves.png")
    plot_learning_curves(history, plot_path)
    print(f"\nSaved learning curves to {plot_path}")
    
    # Test Evaluation (optional)
    if args.test_csv:
        print("\nLoading test dataset...")
        test_dataset = ProteinDataset(args.test_csv, num_classes=args.num_classes)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # Load best model checkpoint
        print(f"Loading best model weights from {best_model_path} for testing...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Evaluation | Loss: {test_loss:.4f} - Accuracy: {test_acc*100:.2f}%")
        
        # Save test results
        with open(os.path.join(args.save_dir, "test_results.txt"), "w", encoding="utf-8") as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc*100:.2f}%\n")


if __name__ == "__main__":
    main()
