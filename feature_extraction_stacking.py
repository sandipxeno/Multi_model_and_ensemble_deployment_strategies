import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import joblib
from smallCNN_stacking import SmallCNN, device
from bagging import LightCNN

# --- Model Loading (Pre-trained models, no training happens here) ---
print("Loading pre-trained models...")

# 1. Load LightCNN (Bagging model)
lightcnn_model = LightCNN().to(device)
lightcnn_model.load_state_dict(torch.load('D:/Prodigal-5/Models/bagging/bagging_model_0.pth'))
lightcnn_model.eval()
print("LightCNN loaded (for predictions only).")

# 2. Load XGBoost (Boosting model)
xgb_model = joblib.load('D:/Prodigal-5/Models/boosting/boosting_xgboost_model.pkl')
print("XGBoost loaded (for predictions only).")

# 3. Load SmallCNN (Stacking model)
small_cnn = SmallCNN().to(device)
small_cnn.load_state_dict(torch.load('D:/Prodigal-5/Models/stacking/stacking_smallcnn.pth'))
small_cnn.eval()
print("SmallCNN loaded (for predictions only).")

# --- Data Loading ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),   # 🔥 CHANGED from (64, 64) to (32, 32)
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root='cifar10/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
print(f"Data loaded. Total batches: {len(train_loader)}")

# --- Generate Stacking Features ---
stacked_features = []
stacked_labels = []

print("Generating stacking features...")

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)

        # 1. Get LightCNN predictions
        lightcnn_probs = torch.softmax(lightcnn_model(images), dim=1).cpu().numpy()

        # 2. Get SmallCNN predictions
        smallcnn_probs = torch.softmax(small_cnn(images), dim=1).cpu().numpy()

        # 3. Get XGBoost predictions (fixed)
        xgb_inputs = torch.mean(images, dim=1, keepdim=True)
        xgb_inputs = torch.nn.functional.interpolate(xgb_inputs, size=(64, 64), mode='bilinear', align_corners=False)
        xgb_features = xgb_inputs.view(xgb_inputs.size(0), -1).cpu().numpy()
        xgb_probs = xgb_model.predict_proba(xgb_features)

        # Combine predictions
        batch_features = np.concatenate([lightcnn_probs, smallcnn_probs, xgb_probs], axis=1)
        stacked_features.append(batch_features)
        stacked_labels.append(labels.numpy())

        if (batch_idx + 1) % 20 == 0:
            print(f"Processed {batch_idx + 1}/{len(train_loader)} batches")


# --- Save Stacking Features ---
stacked_features = np.vstack(stacked_features)
stacked_labels = np.hstack(stacked_labels)

np.save('stacking_features.npy', stacked_features)
np.save('stacking_labels.npy', stacked_labels)

print("Stacking features and labels saved successfully!")
print(f"Stacking features shape: {stacked_features.shape}")
print(f"Labels shape: {stacked_labels.shape}")
