import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
 
class LightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # First convolution layer
            nn.ReLU(),
            nn.MaxPool2d(2),  # First max-pooling layer
            nn.Conv2d(32, 64, 3, padding=1),  # Second convolution layer
            nn.ReLU(),
            nn.MaxPool2d(2),  # Second max-pooling layer
            nn.Flatten()  # Flatten the output
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)  # Adjust to 4096

    def forward(self, x):
        x = self.features(x)
        print(x.shape)  # Print the shape of the flattened tensor
        x = x.view(x.size(0), -1)  # Flatten the tensor (batch_size, -1) to match classifier input
        print(x.shape)  # Check the flattened shape
        x = self.classifier(x)
        return x


# --- Training code (only runs when file is executed directly) ---
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(root='cifar10/train', transform=transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_one_model(train_loader, model_idx):
        model = LightCNN(num_classes=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(5):
            model.train()
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Model {model_idx}, Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f'bagging_model_{model_idx}.pth')

    num_models = 5
    for i in range(num_models):
        indices = random.sample(range(len(train_dataset)), int(0.8 * len(train_dataset)))
        subset = Subset(train_dataset, indices)
        train_loader = DataLoader(subset, batch_size=64, shuffle=True)
        train_one_model(train_loader, i)