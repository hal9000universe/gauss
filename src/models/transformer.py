#  Copyright (c) 2023. Benjamin Schoofs

import torch
import torch.nn as nn
import tokenizers

from typing import Optional


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, x.size(1)]
        return self.dropout(x)


class MathFormer(nn.Module):
    """Math Former Network.

    Args:
        embed_dim (int): embedding dimension
        dim_feedforward (int): dimension of the feedforward network
        num_heads (int): number of heads
        num_layers (int): number of layers
        tokenizer (tokenizers.Tokenizer): tokenizer
    """
    _embedding: nn.Embedding
    _positional_encoding: PositionalEncoding
    _transformer: nn.TransformerEncoder
    _prediction_head: nn.Linear
    _normalization: nn.LayerNorm
    _tokenizer: tokenizers.Tokenizer

    def __init__(self,
                 embed_dim: int,
                 dim_feedforward: int,
                 num_heads: int,
                 num_layers: int,
                 tokenizer: tokenizers.Tokenizer):
        super().__init__()
        # check well-defined
        assert embed_dim % num_heads == 0
        # save parameters
        self._embed_dim = embed_dim
        self._dim_feedforward = dim_feedforward
        self._num_heads = num_heads
        self._num_layers = num_layers
        # create embedding
        self._embedding: nn.Embedding = nn.Embedding(tokenizer.get_vocab_size(), embed_dim,
                                                     padding_idx=tokenizer.token_to_id("[PAD]"))
        # create positional encoding
        self._positional_encoding = PositionalEncoding(embed_dim)
        # create transformer
        self._transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                dim_feedforward=dim_feedforward,
                nhead=num_heads,
                batch_first=True),
            num_layers=num_layers,
        )
        # create normalization
        self._normalization = nn.LayerNorm(embed_dim)
        # create prediction head
        self._prediction_head = nn.Linear(embed_dim, tokenizer.get_vocab_size())
        # create tokenizer attribute
        self._tokenizer = tokenizer

    @property
    def embed_dim(self) -> int:
        """Returns the embedding dimension."""
        return self._embed_dim

    @property
    def dim_feedforward(self) -> int:
        """Returns the dimension of the feedforward network."""
        return self._dim_feedforward

    @property
    def num_heads(self) -> int:
        """Returns the number of heads."""
        return self._num_heads

    @property
    def num_layers(self) -> int:
        """Returns the number of layers."""
        return self._num_layers

    @property
    def tokenizer(self) -> tokenizers.Tokenizer:
        """Returns the tokenizer."""
        return self._tokenizer

    def forward(
            self,
            x: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): input tensor, shape (batch_size, seq_len)
            targets (Optional[torch.Tensor], optional): targets shape (batch_size, target_len). Defaults to None.
            src_key_padding_mask (Optional[torch.Tensor], optional): shape (batch_size, seq_len). Defaults to None.

        Returns:
            torch.Tensor: output tensor
        """
        x = self._embedding(x)
        x = self._positional_encoding(x)
        x = self._transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self._normalization(x)
        logits = self._prediction_head(x[:, -1, :])
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits, targets,
                                                     ignore_index=self._tokenizer.token_to_id("[PAD]"))
            return loss
        else:
            return logits