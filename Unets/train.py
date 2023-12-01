import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
from utils import data_loading
from utils import transform
from unet.unet_model import UNet
from basic_config import root_dir

# Set random seed for reproducibility
torch.manual_seed(0)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
num_epochs = 2

# Set transforms
train_transform = transform.SegmentationTrainingTransform()
val_transform = transform.SegmentationValidationTransform()

# Initialize dataset and dataloader
train_dataset = data_loading.VideoFrameDataset(root_dir=root_dir, subset='train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# dimension of the input to the model
input_channel = 33  # 3 RGB channels per frame, 11 frames
output_channel = 22  # 22 classes for segmentation

# Initialize the UNet model
model = UNet(input_channel, output_channel)  # Replace with your UNet architecture
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for frames, masks in train_loader:
        optimizer.zero_grad()
        # concatenate frames along the channel axis
        frames = torch.cat(frames, dim=1).to(device)
        output = model(frames)
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
