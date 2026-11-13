# PyTorch Custom Dataset & DataLoader

A custom dataset inherits from `torch.utils.data.Dataset` and implements `__len__` and `__getitem__`.

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```
