# PyTorch CUDA & Device Management

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs, targets = inputs.to(device), targets.to(device)
```
