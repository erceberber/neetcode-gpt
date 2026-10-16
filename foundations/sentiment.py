import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embed = nn.Embedding(vocabulary_size, 16)
        self.ly1 = nn.Linear(16, 1)
        self.out = nn.Sigmoid()
        pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        y_hat = self.embed(x)
        y_hat = torch.mean(y_hat, axis=1)
        y_hat = self.ly1(y_hat)
        y_hat = self.out(y_hat)
        return torch.round(y_hat, decimals=4)
        pass
