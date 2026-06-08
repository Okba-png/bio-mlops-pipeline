"""
Data Processing Module for Protein Secondary Structure Prediction.

This module contains functions to clean raw protein sequence data, filter out
non-standard sequences or sequences exceeding a specific length, and split the
data into train, validation, and test sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Standard 20 amino acids
VALID_AMINO_ACIDS = set("ARNDCQEGHILKMFPSTWYV")

def clean_data(df: pd.DataFrame, max_len: int = 512) -> pd.DataFrame:
    """
    Cleans the protein sequence DataFrame.
    
    Filters based on:
    - Sequence length less than or equal to max_len.
    - Matching lengths between sequence and secondary structure labels.
    - Presence of non-standard amino acids.
    """
    if df.empty:
        return df

    # Drop rows with missing values in key columns
    df = df.dropna(subset=['seq', 'sst3'])

    # Standardize string columns to uppercase
    df = df.copy()
    df['seq'] = df['seq'].str.upper()
    df['sst3'] = df['sst3'].str.upper()
    if 'sst8' in df.columns:
        df['sst8'] = df['sst8'].str.upper()

    # 1. Filter by maximum length
    df = df[df['seq'].str.len() <= max_len]

    # 2. Ensure sequence and secondary structure label lengths match
    df = df[df['seq'].str.len() == df['sst3'].str.len()]
    if 'sst8' in df.columns:
        df = df[df['seq'].str.len() == df['sst8'].str.len()]

    # 3. Filter out non-standard amino acids if flag has_nonstd_aa is present,
    # otherwise manually check sequence characters.
    if 'has_nonstd_aa' in df.columns:
        # In Aladdin's dataset this is has_nonstd_aa or has_nonstd_aa == True
        df = df[df['has_nonstd_aa'] == False]
    else:
        # Manual check
        is_standard = df['seq'].apply(lambda x: all(c in VALID_AMINO_ACIDS for c in x))
        df = df[is_standard]

    return df

def split_data(
    df: pd.DataFrame, 
    test_size: float = 0.1, 
    val_size: float = 0.1, 
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train, validation, and test sets.
    """
    if len(df) < 3:
        raise ValueError("Dataset too small to split into train, val, and test.")

    # Calculate test split
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Calculate relative validation size for the remaining train_val set
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=adjusted_val_size, random_state=random_state
    )

    return train_df, val_df, test_df

def prepare_data(
    csv_path: str,
    output_dir: str,
    max_len: int = 512,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42
) -> tuple[str, str, str]:
    """
    Reads, cleans, splits, and saves the data to the output directory.
    Returns the paths to the saved train, val, and test CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading raw data from {csv_path}...")
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    print("Cleaning data...")
    cleaned_df = clean_data(df, max_len=max_len)
    cleaned_count = len(cleaned_df)
    print(f"Cleaned dataset: {cleaned_count}/{initial_count} sequences kept.")
    
    print("Splitting data...")
    train_df, val_df, test_df = split_data(
        cleaned_df, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved splits to {output_dir}:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")
    
    return train_path, val_path, test_path
