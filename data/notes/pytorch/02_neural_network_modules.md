# PyTorch Neural Network Modules (nn.Module)

All neural network models in PyTorch inherit from `nn.Module`.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```
