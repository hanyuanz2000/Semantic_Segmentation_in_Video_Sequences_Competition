# Import necessary libraries
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from pathlib import Path
import os

# Your custom modules
from unet.unet_model import UNet
from utils import data_loading, transform
from basic_config import root_dir
from utils.dice_score import dice_loss
from evaluate import evaluate  # Make sure this is properly defined
from datetime import datetime

# Function to parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # Add your custom arguments here
    return parser.parse_args()

# Function to train the model
def train_model(
        model, 
        device, 
        epochs: int = 1, 
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-8,
        momentum: float = 0.9,
        graient_clipping: float = 1.0,
        save_checkpoint: bool = True,
        amp: bool = False,
    ):

    # set experiment name
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    experiment_name = f'Unet_{dt_string}'
    print(f'Experiment name: {experiment_name}')
    # Initialize Weights & Biases logging
    experiment = wandb.init(project='VideoFrameSegmentation_with_Unet', name=experiment_name, config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "gradient_clipping": graient_clipping,
    })

    # Initialize dataset
    train_transform = transform.SegmentationTrainingTransform()
    val_transform = transform.SegmentationValidationTransform()

    train_set = data_loading.VideoFrameDataset(root_dir=root_dir, subset='train', transform=train_transform)
    val_set = data_loading.VideoFrameDataset(root_dir=root_dir, subset='val', transform=val_transform)
    n_train, n_val = len(train_set), len(val_set)

    # Initialize data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    logging.info(f'Initializing training on {len(train_set)} training images and {len(val_set)} validation images')
    logging.info(f'Using device {device}')
    logging.info(f'Image size: {train_set[0][0][0].size()}')

    # Initialize optimizer, loss function and learning rate scheduler
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                frames, true_masks = batch
                frames = torch.cat(frames, dim=1).to(device, dtype=torch.float32)
                if true_masks.ndim > 3:  # This means masks are one-hot encoded
                    true_masks = torch.argmax(true_masks, dim=1)
                true_masks = true_masks.to(device,  dtype=torch.long)

                masks_pred = model(frames)
                # check dimensions
                # print(f'masks_pred size: {masks_pred.size()}')
                # print(f'true_masks size: {true_masks.size()}')

                loss = criterion(masks_pred, true_masks)

                # Apply softmax to masks_pred to get probabilities
                masks_pred_softmax = F.softmax(masks_pred, dim=1)
                # Convert true_masks to one-hot encoding
                true_masks_one_hot = F.one_hot(true_masks, num_classes=model.n_classes)
                true_masks_one_hot = true_masks_one_hot.permute(0, 3, 1, 2)  # Change shape to [batch_size, n_classes, H, W]

                if model.n_classes > 1:  # If using multiclass, include Dice loss
                    loss_dice = dice_loss(masks_pred_softmax, true_masks_one_hot, multiclass=True)
                    loss += loss_dice

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(frames.shape[0])
                epoch_loss += loss.item()

                experiment.log({"loss": loss.item()})

                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Validation part after each epoch
            val_score = evaluate(model, val_loader, device, amp)
            scheduler.step(val_score)

            logging.info(f'Epoch finished! Loss: {epoch_loss / len(train_loader)} Validation Dice Score: {val_score}')
            wandb.log({"epoch": epoch, "loss": epoch_loss / len(train_loader), "validation score": val_score})

            if save_checkpoint:
                Path('./checkpoints').mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), f'./checkpoints/checkpoint_epoch{epoch}.pth')

    print("Training completed")

# Main script execution
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
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

    logging.info(f'Using device {device}')

    # Initialize model
    # dimension of the input to the model
    input_channel = 33  # 3 RGB channels per frame, 11 frames
    output_channel = 22  # 22 classes for segmentation

    # Initialize the UNet model
    model = UNet(input_channel, output_channel)
    model.to(device)

    train_model(
            model=model,
            device=device
        )

    # try:
    #     train_model(
    #         model=model,
    #         device=device
    #     )
    # except Exception as e:
    #     logging.error(f'Error during training: {e}')






