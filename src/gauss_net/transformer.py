import torch
import torch.nn as nn
import tokenizers

from typing import Optional

from src.data_engine.data_pipe import get_tokenizer


class GaussNet(nn.Module):
    """Gaussian Elimination Network."""
    _embedding: nn.Embedding
    _transformer: nn.TransformerEncoder
    _prediction_head: nn.Linear

    def __init__(self, embedding: nn.Embedding, transformer: nn.TransformerEncoder, prediction_head: nn.Linear):
        super().__init__()
        self._embedding = embedding
        self._transformer = transformer
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
        x = self._transformer(x, src_key_padding_mask=src_key_padding_mask)
        logits = self._prediction_head(x[:, -1, :])
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits, targets,
                                                     ignore_index=get_tokenizer().token_to_id("[PAD]"))
            return loss
        else:
            return logits


def create_gauss_net(embed_dim: int = 64, dim_feedforward: int = 512) -> GaussNet:
    """Creates the model.
    
    Returns:
        GaussNet: model
    """
    # get tokenizer
    tokenizer: tokenizers.Tokenizer = get_tokenizer()
    # create embedding
    embedding: nn.Embedding = nn.Embedding(tokenizer.get_vocab_size(), embed_dim,
                                           padding_idx=tokenizer.token_to_id("[PAD]"))
    # create transformer
    transformer: nn.TransformerEncoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=embed_dim,
            dim_feedforward=dim_feedforward,
            nhead=4,
            batch_first=True),
        num_layers=5
    )
    # create prediction head
    prediction_head: nn.Linear = nn.Linear(embed_dim, tokenizer.get_vocab_size())
    # create model
    model = GaussNet(embedding, transformer, prediction_head)
    return model
