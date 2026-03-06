import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as snsp
from Train_Model import train_data_loader, test_data_loader, device
from collections import Counter

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 5)

model.load_state_dict(torch.load("Datasets/landuse_resnet18.pth"))
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_data_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


Accuracy = accuracy_score(all_labels, all_predictions)
print("Model Accuracy", Accuracy)  

f1 = f1_score(all_labels, all_predictions, average="weighted")
print("F1 Score:", f1)

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(6,5))
snsp.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()