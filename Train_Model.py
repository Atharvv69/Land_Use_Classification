from Train_Test_Split import x_train, x_test, y_train, y_test
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

valid_classes = [10,20,30,40,50]


train_pairs = [(x,y) for x,y in zip(x_train,y_train) if y in valid_classes]
x_train = [x for x,y in train_pairs]
y_train = [y for x,y in train_pairs]

test_pairs = [(x,y) for x,y in zip(x_test,y_test) if y in valid_classes]
x_test = [x for x,y in test_pairs]
y_test = [y for x,y in test_pairs]

class_mappings = {
    10:0,
    20:1,
    30:2,
    40:3,
    50:4
}

y_train_mapped = [class_mappings[i] for i in y_train]
y_test_mapped = [class_mappings[i] for i in y_test]


class Land_Use_Dataset(Dataset):

    def __init__(self, image_names, labels, folder, transform= None):
        self.image_names = image_names
        self.labels = labels
        self.folder = folder
        self.transform = transform
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        
        img_path = os.path.join(self.folder, self.image_names[index])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        label = self.labels[index]

        return image, label

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


train_dataset = Land_Use_Dataset(
    x_train,
    y_train_mapped,
    "Datasets/Valid_Images",
    transform
)

test_dataset = Land_Use_Dataset(
    x_test,
    y_test_mapped,
    "Datasets/Valid_Images",
    transform
)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model = resnet18(weights=ResNet18_Weights.DEFAULT)


model.fc = nn.Linear(model.fc.in_features, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    num_epochs = 8

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        for image, labels in train_data_loader:
            images = image.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
     
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


    torch.save(model.state_dict(), "Datasets/landuse_resnet18.pth")


