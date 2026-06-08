"""
PyTorch Dataset and Dataloader utilities for Protein Secondary Structure Prediction.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Amino acid alphabet: 20 standard AAs + <PAD> (0) and <UNK> (21)
AA_ALPHABET = ["<PAD>"] + list("ARNDCQEGHILKMFPSTWYV") + ["<UNK>"]
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}
IDX_TO_AA = {idx: aa for idx, aa in enumerate(AA_ALPHABET)}

# Secondary structure alphabet for Q3: H (Helix), E (Strand), C (Coil)
Q3_ALPHABET = list("HEC")
Q3_TO_IDX = {ss: idx for idx, ss in enumerate(Q3_ALPHABET)}
IDX_TO_Q3 = {idx: ss for idx, ss in enumerate(Q3_ALPHABET)}

# Secondary structure alphabet for Q8 (if needed in future)
Q8_ALPHABET = list("GHIBESTC")
Q8_TO_IDX = {ss: idx for idx, ss in enumerate(Q8_ALPHABET)}
IDX_TO_Q8 = {idx: ss for idx, ss in enumerate(Q8_ALPHABET)}


def encode_string(text: str, vocab: dict, default_val: int) -> list[int]:
    """
    Encodes a string of characters into a list of integers based on a vocabulary mapping.
    """
    return [vocab.get(char, default_val) for char in text]


class ProteinDataset(Dataset):
    """
    Custom PyTorch Dataset for protein sequences and secondary structures.
    """
    def __init__(self, csv_path: str, num_classes: int = 3):
        """
        Args:
            csv_path: Path to the processed CSV file.
            num_classes: Number of structure classes (3 for Q3, 8 for Q8).
        """
        self.df = pd.read_csv(csv_path)
        self.num_classes = num_classes
        
        # Ensure sequence columns are strings
        self.df['seq'] = self.df['seq'].astype(str)
        self.df['sst3'] = self.df['sst3'].astype(str)
        if 'sst8' in self.df.columns:
            self.df['sst8'] = self.df['sst8'].astype(str)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        seq_str = row['seq']
        
        # Encode inputs (amino acids)
        seq_encoded = encode_string(seq_str, AA_TO_IDX, AA_TO_IDX["<UNK>"])
        
        # Encode targets (secondary structures)
        if self.num_classes == 3:
            struct_str = row['sst3']
            struct_encoded = encode_string(struct_str, Q3_TO_IDX, Q3_TO_IDX['C'])
        elif self.num_classes == 8:
            if 'sst8' not in self.df.columns:
                raise ValueError("Dataset does not contain 'sst8' column for Q8 prediction.")
            struct_str = row['sst8']
            struct_encoded = encode_string(struct_str, Q8_TO_IDX, Q8_TO_IDX['C'])
        else:
            raise ValueError(f"Invalid num_classes: {self.num_classes}. Supported: 3 or 8.")
            
        return (
            torch.tensor(seq_encoded, dtype=torch.long),
            torch.tensor(struct_encoded, dtype=torch.long)
        )


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to pad variable-length sequences.
    
    Args:
        batch: List of tuples (seq_tensor, struct_tensor).
        
    Returns:
        padded_seqs: Tensor of shape (batch_size, max_seq_len) padded with 0.
        padded_structs: Tensor of shape (batch_size, max_seq_len) padded with -100.
        src_key_padding_mask: Bool Tensor of shape (batch_size, max_seq_len)
                              where True represents padded positions.
    """
    sequences, structures = zip(*batch)
    
    # Pad inputs with 0 (idx of <PAD>)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Pad labels with -100 (ignored by PyTorch's nn.CrossEntropyLoss)
    padded_structs = pad_sequence(structures, batch_first=True, padding_value=-100)
    
    # Create padding mask: True where sequences are padded (equal to 0)
    src_key_padding_mask = (padded_seqs == 0)
    
    return padded_seqs, padded_structs, src_key_padding_mask
