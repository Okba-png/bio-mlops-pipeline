"""
Unit tests for data_processing.py.
"""

import pandas as pd
import pytest
from src.data_processing import clean_data, split_data


def test_clean_data_normal():
    # Arrange
    data = {
        'seq': ['MKTLLIL', 'MVLSEGEW', 'ACDEF'],
        'sst3': ['CHHHHHH', 'CEEEEEEE', 'CCCCC'],
        'has_nonstd_aa': [False, False, False]
    }
    df = pd.DataFrame(data)
    
    # Act
    cleaned = clean_data(df, max_len=10)
    
    # Assert
    assert len(cleaned) == 3
    assert list(cleaned['seq']) == ['MKTLLIL', 'MVLSEGEW', 'ACDEF']


def test_clean_data_length_filter():
    # Arrange
    data = {
        'seq': ['MKTLLIL', 'MVLSEGEW', 'ACDEF'],
        'sst3': ['CHHHHHH', 'CEEEEEEE', 'CCCCC'],
        'has_nonstd_aa': [False, False, False]
    }
    df = pd.DataFrame(data)
    
    # Act: max_len of 6 should drop MKTLLIL (7) and MVLSEGEW (8), leaving ACDEF (5)
    cleaned = clean_data(df, max_len=6)
    
    # Assert
    assert len(cleaned) == 1
    assert list(cleaned['seq']) == ['ACDEF']


def test_clean_data_mismatched_length():
    # Arrange
    data = {
        'seq': ['MKTLLIL', 'MVLSEGEW'],
        'sst3': ['CHHH', 'CEEEEEEE'],  # MKTLLIL has 7 AAs but only 4 sst3 states
        'has_nonstd_aa': [False, False]
    }
    df = pd.DataFrame(data)
    
    # Act
    cleaned = clean_data(df)
    
    # Assert
    assert len(cleaned) == 1
    assert list(cleaned['seq']) == ['MVLSEGEW']


def test_clean_data_nonstd_amino_acids():
    # Arrange
    data = {
        'seq': ['MKTLLIL', 'MKTLLXIL'],  # Second contains X (non-standard)
        'sst3': ['CHHHHHH', 'CHHHHHHH'],
        'has_nonstd_aa': [False, True]
    }
    df = pd.DataFrame(data)
    
    # Act
    cleaned = clean_data(df)
    
    # Assert
    assert len(cleaned) == 1
    assert list(cleaned['seq']) == ['MKTLLIL']


def test_clean_data_manual_nonstd_check():
    # Arrange - df without 'has_nonstd_aa' column, should use manual check
    data = {
        'seq': ['MKTLLIL', 'MKTLLXIL'],
        'sst3': ['CHHHHHH', 'CHHHHHHH'],
    }
    df = pd.DataFrame(data)
    
    # Act
    cleaned = clean_data(df)
    
    # Assert
    assert len(cleaned) == 1
    assert list(cleaned['seq']) == ['MKTLLIL']


def test_split_data():
    # Arrange
    data = {
        'seq': [f"MKTLLIL{i}" for i in range(10)],
        'sst3': [f"CHHHHHH{i}" for i in range(10)],
    }
    df = pd.DataFrame(data)
    
    # Act
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.2, random_state=42)
    
    # Assert
    # 10 samples: test_size=0.2 (2 samples), val_size=0.2 (2 samples), train_size=6 samples
    assert len(train_df) == 6
    assert len(val_df) == 2
    assert len(test_df) == 2
    
    # Ensure no overlaps
    assert set(train_df['seq']).isdisjoint(set(val_df['seq']))
    assert set(train_df['seq']).isdisjoint(set(test_df['seq']))
    assert set(val_df['seq']).isdisjoint(set(test_df['seq']))