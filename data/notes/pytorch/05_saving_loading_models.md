# Saving & Loading Models in PyTorch

Always save `state_dict` (recommended) rather than the whole model object.

```python
# Save
torch.save(model.state_dict(), "model_weights.pt")

# Load
model = MLP(10, 32, 2)
model.load_state_dict(torch.load("model_weights.pt"))
model.eval()
```
