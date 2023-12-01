import argparse
import logging
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet.unet_model import UNet
from utils import data_loading, transform
from basic_config import root_dir

# Set random seed for reproducibility
torch.manual_seed(0)

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA on device: {torch.cuda.get_device_name(0)}")
# Check for MPS availability
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
# Fallback to CPU if neither CUDA nor MPS is available
else:
    device = torch.device("cpu")
    print("CUDA and MPS not available, using CPU")

print(f'Using device: {device}')

# Set hyperparameters
num_epochs = 1
batch_size = 8

# Set transforms
train_transform = transform.SegmentationTrainingTransform()
val_transform = transform.SegmentationValidationTransform()

# Initialize dataset and dataloader
train_dataset = data_loading.VideoFrameDataset(root_dir=root_dir, subset='train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = data_loading.VideoFrameDataset(root_dir=root_dir, subset='val', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# dimension of the input to the model
input_channel = 33  # 3 RGB channels per frame, 11 frames
output_channel = 22  # 22 classes for segmentation

# Initialize the UNet model
model = UNet(input_channel, output_channel)  # Replace with your UNet architecture
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Progress bar (tqdm)
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for i, batch in enumerate(train_bar):
        frames, masks = batch  # Unpack the batch
        optimizer.zero_grad()
        
        # concatenate frames along the channel axis
        frames = torch.cat(frames, dim=1).to(device, dtype=torch.float32)
        if masks.ndim > 3:  # This means masks are one-hot encoded
            masks = torch.argmax(masks, dim=1)
        masks = masks.to(device,  dtype=torch.long)
        output = model(frames)

        # print(f'Input size: {frames.size()}')
        # print(f'Output size: {output.size()}')
        # print(f'Mask size: {masks.size()}')

        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()

        # update the running loss
        running_loss += loss.item()
        # Use i+1 as the denominator to calculate the average loss
        train_bar.set_postfix(loss=(running_loss / (i + 1)))
        
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

print("Training completed")
