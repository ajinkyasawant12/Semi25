import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DPTForDepthEstimation, DPTImageProcessor

# Define your custom dataset
class CustomDepthDataset(Dataset):
    def __init__(self, image_paths, depth_paths, transform=None):
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        depth = Image.open(self.depth_paths[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
        return image, depth

# Define your training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, depths in dataloader:
            images = images.to(device)
            depths = depths.to(device)

            optimizer.zero_grad()
            outputs = model(images).predicted_depth
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model

# Load your data
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
depth_paths = ["path/to/depth1.png", "path/to/depth2.png", ...]
transform = transforms.Compose([transforms.Resize((480, 640)), transforms.ToTensor()])
dataset = CustomDepthDataset(image_paths, depth_paths, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, criterion, and optimizer
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=25)