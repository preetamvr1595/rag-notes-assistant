# Top PyTorch Interview Questions & Answers

### Q1: Why do we call optimizer.zero_grad()?
PyTorch accumulates gradients by default. Without zeroing, gradients from previous iterations add up.

### Q2: What does model.eval() do?
Sets BatchNorm and Dropout layers to evaluation mode so they do not update running stats or drop units during inference.
