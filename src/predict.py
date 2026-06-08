"""
Inference script for predicting protein secondary structure from amino acid sequence.
"""

import argparse
import os
import torch
import torch.nn as nn

from src.dataset import AA_TO_IDX, IDX_TO_Q3, IDX_TO_Q8, AA_ALPHABET
from src.model import ProteinTransformer


def predict_sequence(
    model: nn.Module, 
    sequence: str, 
    num_classes: int = 3, 
    device: torch.device = torch.device("cpu")
) -> str:
    """
    Predicts the secondary structure for a single amino acid sequence.
    
    Args:
        model: Trained ProteinTransformer model.
        sequence: Raw amino acid sequence string (e.g. "MKTLLIL").
        num_classes: Number of structure classes (3 for Q3, 8 for Q8).
        device: Torch device.
        
    Returns:
        Predicted secondary structure string.
    """
    model.eval()
    
    # 1. Clean and tokenize sequence
    sequence = sequence.upper().strip()
    encoded = [AA_TO_IDX.get(char, AA_TO_IDX["<UNK>"]) for char in sequence]
    
    # 2. Convert to tensor and add batch dimension (1, seq_len)
    seq_tensor = torch.tensor([encoded], dtype=torch.long).to(device)
    
    # 3. Model forward pass
    with torch.no_grad():
        logits = model(seq_tensor)  # (1, seq_len, num_classes)
        predictions = logits.argmax(dim=-1).squeeze(0)  # (seq_len)
        
    # 4. Map back to secondary structure labels
    idx_to_label = IDX_TO_Q3 if num_classes == 3 else IDX_TO_Q8
    predicted_labels = [idx_to_label[idx.item()] for idx in predictions]
    
    return "".join(predicted_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict protein secondary structure using a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--sequence", type=str, required=True, help="Amino acid sequence (e.g., MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")
        
    # Load checkpoint
    print(f"Loading checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Retrieve hyperparameters from checkpoint to build the model structure
    saved_args = checkpoint['args']
    print(f"Loaded model configuration: d_model={saved_args.d_model}, nhead={saved_args.nhead}, num_layers={saved_args.num_layers}, num_classes={saved_args.num_classes}")
    
    model = ProteinTransformer(
        vocab_size=len(AA_ALPHABET),
        d_model=saved_args.d_model,
        nhead=saved_args.nhead,
        num_layers=saved_args.num_layers,
        dim_feedforward=saved_args.dim_feedforward,
        num_classes=saved_args.num_classes
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Predict
    prediction = predict_sequence(
        model=model,
        sequence=args.sequence,
        num_classes=saved_args.num_classes,
        device=device
    )
    
    print("\n--- Prediction Results ---")
    print(f"Sequence:   {args.sequence.upper()}")
    print(f"Prediction: {prediction}")
    print("--------------------------")


if __name__ == "__main__":
    main()
