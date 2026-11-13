# PyTorch Training Loop Standard Pattern

The canonical 5-step training loop:
1. Zero gradients: `optimizer.zero_grad()`
2. Forward pass: `outputs = model(inputs)`
3. Compute loss: `loss = criterion(outputs, targets)`
4. Backward pass: `loss.backward()`
5. Update weights: `optimizer.step()`
