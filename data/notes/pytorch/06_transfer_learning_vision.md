# Transfer Learning with torchvision

Pre-trained models on ImageNet (ResNet, EfficientNet) can be fine-tuned or used as feature extractors.

```python
import torchvision.models as models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # Freeze backbone

model.fc = nn.Linear(model.fc.in_features, 10)  # Replace classification head
```
