"""
Unit tests for model.py and dataset.py.
"""

import os
import tempfile
import pandas as pd
import pytest
import torch

from src.dataset import ProteinDataset, collate_fn, AA_TO_IDX, Q3_TO_IDX
from src.model import ProteinTransformer


def test_protein_transformer_forward():
    # Arrange
    vocab_size = 22
    d_model = 64
    nhead = 2
    num_layers = 1
    num_classes = 3
    
    model = ProteinTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=128,
        num_classes=num_classes
    )
    
    batch_size = 4
    seq_len = 10
    # Inputs: batch_size x seq_len (values in [0, vocab_size-1])
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Act
    outputs = model(inputs)
    
    # Assert
    assert outputs.shape == (batch_size, seq_len, num_classes)


def test_protein_transformer_with_mask():
    # Arrange
    vocab_size = 22
    d_model = 64
    nhead = 2
    num_layers = 1
    num_classes = 3
    
    model = ProteinTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes
    )
    
    # Create inputs where some values are 0 (padding)
    inputs = torch.tensor([
        [1, 2, 3, 0, 0],
        [4, 5, 0, 0, 0]
    ])  # shape (2, 5)
    
    mask = (inputs == 0)  # True where padded (True means ignored)
    
    # Act
    outputs = model(inputs, src_key_padding_mask=mask)
    
    # Assert
    assert outputs.shape == (2, 5, num_classes)
    # Output at padded index should be computable without NaNs
    assert not torch.isnan(outputs).any()


def test_collate_fn():
    # Arrange
    # List of tuples: (seq_tensor, struct_tensor)
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2])),        # length 3
        (torch.tensor([4, 5]), torch.tensor([1, 2])),              # length 2
        (torch.tensor([6, 7, 8, 9]), torch.tensor([2, 0, 1, 2]))   # length 4
    ]
    
    # Act
    padded_seqs, padded_structs, masks = collate_fn(batch)
    
    # Assert
    # Batch size = 3, Max length = 4
    assert padded_seqs.shape == (3, 4)
    assert padded_structs.shape == (3, 4)
    assert masks.shape == (3, 4)
    
    # Check padding values
    # seqs pad value should be 0
    assert padded_seqs[1, 2].item() == 0
    assert padded_seqs[1, 3].item() == 0
    
    # structs pad value should be -100
    assert padded_structs[1, 2].item() == -100
    assert padded_structs[1, 3].item() == -100
    
    # Check mask values: True where padded
    assert masks[0, 3].item() is True
    assert masks[1, 2].item() is True
    assert masks[1, 3].item() is True
    assert masks[2, 3].item() is False  # Last sequence has length 4, no padding


def test_protein_dataset():
    # Arrange
    data = {
        'seq': ['MKT', 'MVLS'],
        'sst3': ['CHH', 'CEEE']
    }
    df = pd.DataFrame(data)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        df.to_csv(csv_path, index=False)
        
        # Act
        dataset = ProteinDataset(csv_path, num_classes=3)
        
        # Assert
        assert len(dataset) == 2
        
        seq_tensor, struct_tensor = dataset[0]
        # 'MKT' -> [AA_TO_IDX['M'], AA_TO_IDX['K'], AA_TO_IDX['T']]
        expected_seq = [AA_TO_IDX['M'], AA_TO_IDX['K'], AA_TO_IDX['T']]
        assert seq_tensor.tolist() == expected_seq
        
        # 'CHH' -> [Q3_TO_IDX['C'], Q3_TO_IDX['H'], Q3_TO_IDX['H']]
        expected_struct = [Q3_TO_IDX['C'], Q3_TO_IDX['H'], Q3_TO_IDX['H']]
        assert struct_tensor.tolist() == expected_struct
