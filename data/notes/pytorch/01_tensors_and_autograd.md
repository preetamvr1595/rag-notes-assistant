# PyTorch Tensors & Autograd Mechanics

Tensors are multi-dimensional arrays with GPU acceleration support. Autograd provides automatic differentiation for all operations on tensors.

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward(torch.tensor([1.0, 1.0]))
print(x.grad)  # 2*x + 3 -> [7.0, 9.0]
```
