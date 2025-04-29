import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Feature Extractor (Small CNN)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.features(x)
        return x

# Transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Load dataset
train_dataset = datasets.ImageFolder(root='cifar10/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = FeatureExtractor().to(device)
model.eval()

# Extract features
all_features = []
all_labels = []

print("Starting feature extraction...")

total_batches = len(train_loader)
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        print(f"Processed batch {batch_idx + 1}/{total_batches}")

# After loop
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Save features and labels
np.save('boosting_features.npy', all_features)
np.save('boosting_labels.npy', all_labels)

print("Features and labels saved successfully!")
