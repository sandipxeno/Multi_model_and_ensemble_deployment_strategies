import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define SmallCNN model (always available for import)
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Device setup (always available for import)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Training code (only runs when file is executed directly) ---
if __name__ == '__main__':
    # Data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(root='cifar10/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model
    small_cnn = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(small_cnn.parameters(), lr=0.001)

    # Train
    num_epochs = 5
    print("Starting training...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = small_cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    torch.save(small_cnn.state_dict(), 'stacking_smallcnn.pth')
    print("Model saved!")