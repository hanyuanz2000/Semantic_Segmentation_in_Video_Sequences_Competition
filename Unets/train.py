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
import torchmetrics

# Your custom modules
from unet.unet_model import UNet
from utils import data_loading
from utils import customized_transform
from basic_config import root_dir
from utils.dice_score import dice_loss
from evaluate import evaluate
from datetime import datetime

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# export PYTORCH_ENABLE_MPS_FALLBACK=1

# Function to parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # Add your custom arguments here
    return parser.parse_args()

# Function to train the model
def train_model(
        model, 
        device, 
        train_transform,
        val_transform,
        model_name: str = 'Unet_Direct_Pred',
        epochs: int = 1, 
        batch_size: int = 4,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-6,
        momentum: float = 0.9,
        gradient_clipping: float = 1.0,
        save_checkpoint: bool = True,
        amp: bool = False,
    ):

    # set experiment name
    now = datetime.now()
    dt_string = now.strftime("%d-%H:%M")
    experiment_name = f'{dt_string}_lr{learning_rate}_bs{batch_size}_wd{weight_decay}_mom{momentum}_gc{gradient_clipping}'
    print(f'Experiment name: {experiment_name}')
    
    # Initialize Weights & Biases logging
    experiment = wandb.init(project='VideoFrameSegmentation', name=experiment_name, config={
        "model_name": model_name,   
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "gradient_clipping": gradient_clipping,
    })

    # load data
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

    # Initialize Jaccard Index
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=model.n_classes).to(device)
    best_jac = 0.0
    best_model_path = None

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_jaccard = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                frames, true_masks = batch
                frames = torch.cat(frames, dim=1).to(device, dtype=torch.float32)
                if true_masks.ndim > 3:  # This means masks are one-hot encoded
                    true_masks = torch.argmax(true_masks, dim=1)
                true_masks = true_masks.to(device,  dtype=torch.long)

                masks_pred = model(frames)

                loss = criterion(masks_pred, true_masks)

                # Apply softmax to masks_pred to get probabilities
                masks_pred_softmax = F.softmax(masks_pred, dim=1)
                # Convert true_masks to one-hot encoding
                true_masks_one_hot = F.one_hot(true_masks, num_classes=model.n_classes)
                true_masks_one_hot = true_masks_one_hot.permute(0, 3, 1, 2)  # Change shape to [batch_size, n_classes, H, W]

                if model.n_classes > 1:  # If using multiclass, include Dice loss
                    loss_dice = dice_loss(masks_pred_softmax, true_masks_one_hot, multiclass=True)
                    loss += loss_dice

                # Calculate Jaccard Index
                mask_pred_argmax = torch.argmax(masks_pred_softmax, dim=1)
                jac_score = jaccard(mask_pred_argmax, true_masks).item()
                epoch_jaccard += jac_score

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(frames.shape[0])
                epoch_loss += loss.item()

                experiment.log({"loss": loss.item()})

                pbar.set_postfix(**{'loss (batch)': loss.item()})
            

            # Validation part after each epoch
            val_jaccard = evaluate(model, val_loader, device, amp)
            scheduler.step(val_jaccard)

            # Log metrics to Weights & Biases
            average_jaccard = epoch_jaccard / len(train_loader)
            experiment.log({"epoch": epoch, "loss": epoch_loss / len(train_loader), "jaccard": average_jaccard, "val_jaccard": val_jaccard})
            logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}, Jaccard: {average_jaccard}, Val_jaccard: {val_jaccard}')
            
            if save_checkpoint and val_jaccard> best_jac:
                # If current model is better, delete the previous best model checkpoint
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                # Save the current model checkpoint
                Path('./checkpoints').mkdir(parents=True, exist_ok=True)
                best_jac = val_jaccard
                best_model_path = f'./checkpoints/checkpoint_epoch{epoch}.pth'
                torch.save(model.state_dict(), best_model_path)
                logging.info(f'New best model saved at epoch {epoch + 1} with Jaccard score: {best_jac}')

    print("Training completed")
    print(f'Best Jaccard score: {best_jac}')
    if best_model_path:
        print(f'Best model saved at {best_model_path}')
    experiment.finish()


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

    train_transform = customized_transform.SegmentationTrainingTransform()
    val_transform = customized_transform.SegmentationValidationTransform()

    # Initialize model
    # dimension of the input to the model
    input_channel = 33  # 3 RGB channels per frame, 11 frames
    output_channel = 22  # 22 classes for segmentation

    # Initialize the UNet model
    model = UNet(input_channel, output_channel)
    model.to(device)

    train_model(
            model=model,
            device=device,
            train_transform=train_transform,
            val_transform=val_transform,
        )

   





