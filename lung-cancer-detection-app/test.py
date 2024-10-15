# Test the model
import numpy as np
import matplotlib.pyplot as plt
import torch
from train import test_loader, device, model, reverse_class_dict

model.load_state_dict(torch.load('model.pth', weights_only=True))

model.eval()
test_running_corrects = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        test_running_corrects += torch.sum(preds == labels.data)

test_acc = test_running_corrects.double() / len(test_loader.dataset)
print(f'Test Accuracy: {test_acc:.4f}')

inputs, labels = next(iter(test_loader))
inputs = inputs.to(device)
labels = labels.to(device)

outputs = model(inputs)
_, preds = torch.max(outputs, 1)

# Convert images to CPU and un normalize
inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)
inputs = np.clip(inputs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

# Plot the images with predictions
plt.figure(figsize=(20, 20))
for i in range(len(inputs)):
    plt.subplot(6, 6, i + 1)
    plt.imshow(inputs[i])
    plt.title(f'Predicted: {reverse_class_dict[preds[i].item()]}\nActual: {reverse_class_dict[labels[i].item()]}', color='black', fontsize=16)
    plt.axis('off')

plt.tight_layout()
plt.show()