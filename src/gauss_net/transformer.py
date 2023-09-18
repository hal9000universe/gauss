import torch
import torch.nn as nn
import tokenizers

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = self._embedding(x)
        x = self._transformer(x)
        x = self._prediction_head(x).squeeze(-1)
        return torch.nn.functional.softmax(x, dim=-1)


def create_gauss_net(embed_dim: int = 100) -> GaussNet:
    """Creates the model.
    
    Returns:
        GaussNet: model
    """
    # get tokenizer
    tokenizer: tokenizers.Tokenizer = get_tokenizer()
    # create embedding
    embedding: nn.Embedding = nn.Embedding(tokenizer.get_vocab_size(), embed_dim, padding_idx=tokenizer.token_to_id("[PAD]"))
    # create transformer
    transformer: nn.TransformerEncoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
        num_layers=5
    )
    # create prediction head
    prediction_head: nn.Linear = nn.Linear(embed_dim, 1)
    # create model
    model = GaussNet(embedding, transformer, prediction_head)
    return model