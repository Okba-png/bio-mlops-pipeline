"""
Transformer Model Architecture for Protein Secondary Structure Prediction.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for injecting sequence order information.
    Designed for batch_first=True tensors.
    """
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings in log space
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        # x is (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ProteinTransformer(nn.Module):
    """
    Transformer Encoder baseline model for sequence-to-sequence protein secondary structure prediction.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layer (padding_idx=0 is used for <PAD>)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=1024, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Linear layer mapping hidden representations to class logits
        self.decoder = nn.Linear(d_model, num_classes)
        
        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor of shape (batch_size, seq_len) containing token indices.
            src_key_padding_mask: Bool Tensor of shape (batch_size, seq_len) 
                                  where True indicates padding tokens.
        
        Returns:
            logits: Tensor of shape (batch_size, seq_len, num_classes)
        """
        # 1. Embed and scale: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # 2. Add Positional Encoding: (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # 3. Pass through Transformer Encoder
        # note: PyTorch's TransformerEncoder expects src_key_padding_mask of shape (batch_size, seq_len)
        # when batch_first=True
        h = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 4. Decode to classes: (batch_size, seq_len, num_classes)
        logits = self.decoder(h)
        return logits
