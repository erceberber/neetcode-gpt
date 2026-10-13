import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.input_layer = nn.Linear(784, 512)
        self.relu_layer = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(512, 10)
        self.activate = nn.Sigmoid()
        
        pass

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        y_hat = self.input_layer(images)
        y_hat = self.relu_layer(y_hat)
        y_hat = self.drop(y_hat)
        y_hat = self.out(y_hat)
        y_hat = self.activate(y_hat)
        return torch.round(y_hat, decimals=4)
        pass
