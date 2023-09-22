#  Copyright (c) 2023. Benjamin Schoofs

import torch
import torch.nn as nn
import tokenizers

from typing import Optional

from src.data_engine.data_pipe import get_tokenizer


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


class GaussNet(nn.Module):
    """Gaussian Elimination Network."""
    _embedding: nn.Embedding
    _positional_encoding: PositionalEncoding
    _transformer: nn.TransformerEncoder
    _prediction_head: nn.Linear

    def __init__(self, embedding: nn.Embedding, positional_encoding: PositionalEncoding,
                 transformer: nn.TransformerEncoder, prediction_head: nn.Linear):
        super().__init__()
        self._embedding = embedding
        self._positional_encoding = positional_encoding
        self._transformer = transformer
        self._normalization = nn.LayerNorm(embedding.embedding_dim)
        self._prediction_head = prediction_head

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
                                                     ignore_index=get_tokenizer().token_to_id("[PAD]"))
            return loss
        else:
            return logits


def create_gauss_net(embed_dim: int = 64, dim_feedforward: int = 512,
                     num_heads: int = 4, num_layers: int = 4) -> GaussNet:
    """Creates the model.
    
    Returns:
        GaussNet: model
    """
    # check well-defined
    assert embed_dim % num_heads == 0
    # get tokenizer
    tokenizer: tokenizers.Tokenizer = get_tokenizer()
    # create embedding
    embedding: nn.Embedding = nn.Embedding(tokenizer.get_vocab_size(), embed_dim,
                                           padding_idx=tokenizer.token_to_id("[PAD]"))
    # create positional encoding
    positional_encoding: PositionalEncoding = PositionalEncoding(embed_dim)
    # create transformer
    transformer: nn.TransformerEncoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=embed_dim,
            dim_feedforward=dim_feedforward,
            nhead=num_heads,
            batch_first=True),
        num_layers=num_layers,
    )
    # create prediction head
    prediction_head: nn.Linear = nn.Linear(embed_dim, tokenizer.get_vocab_size())
    # create model
    model = GaussNet(embedding, positional_encoding, transformer, prediction_head)
    return model
