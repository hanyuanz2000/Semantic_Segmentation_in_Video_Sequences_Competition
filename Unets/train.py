# Import necessary libraries
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb
from pathlib import Path
import os
import torchmetrics

# custom modules
from models.unet_model import UNet
from utils import data_loading
from utils import customized_transform
from utils.dice_score import dice_loss
from datetime import datetime
import numpy as np

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# export PYTORCH_ENABLE_MPS_FALLBACK=1

# Function to parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--amp', type=bool, default=False, help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--model_name', type=str, default='Unet_Direct_Pred', help='Name of the model')
    parser.add_argument('--subset_test', type=bool, default=False, help='Whether to use a subset of the data for testing'),
    parser.add_argument('--root_dir', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data', help='Root directory of the dataset')

    return parser.parse_args()

# Function to train the model
def train_model(
        model, 
        device, 
        train_transform,
        val_transform,
        epochs, 
        batch_size,
        learning_rate,
        weight_decay,
        momentum,
        gradient_clipping,
        amp,
        model_name,
        subset_test,
        root_dir
    ):

    # set project name and initialize experiment 
    project_name = 'Unet_segmenatation' + '_' + model_name
    
    now = datetime.now()
    dt_string = 'Dec' + now.strftime("%d") + '-' + now.strftime("%H:%M")
    config_string = f'lr{learning_rate}_bs{batch_size}_wd{weight_decay}_mom{momentum}_gc{gradient_clipping}'
    experiment_name = f'{dt_string}_{config_string}'
    logging.info(f'Experiment name: {experiment_name}')

    # Create a unique directory for this experiment204
    unique_dir = f'{model_name}_{dt_string}_{config_string}'
    checkpoint_dir = os.path.join('./checkpoints', unique_dir)
    
    # Initialize Weights & Biases logging
    experiment = wandb.init(project=project_name, name=experiment_name, config={
        "model_name": model_name,   
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "gradient_clipping": gradient_clipping,
    })
    
    # load data
    train_set = data_loading.one_to_one_Segmentation_Dataset(root_dir=root_dir, subset='train', transform=train_transform)
    val_set = data_loading.one_to_one_Segmentation_Dataset(root_dir=root_dir, subset='val', transform=val_transform)

    n_train, n_val = len(train_set), len(val_set)

    if subset_test: # Only use a subset of the data for testing
        # Define the size of the subset as 5% of the dataset
        train_subset_size = int(0.03 * n_train)
        val_subset_size = int(0.03 * n_val)

        # Generate random indices for train and validation subsets
        train_subset_indices = np.random.choice(range(n_train), train_subset_size, replace=False)
        val_subset_indices = np.random.choice(range(n_val), val_subset_size, replace=False)

        # Create subsets
        train_subset = Subset(train_set, train_subset_indices)
        val_subset = Subset(val_set, val_subset_indices)
    
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    else: # Load the entire dataset
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

    patience = 6 # Number of epochs to wait before reducing learning rate
    epochs_no_improve = 0 # Number of epochs with no improvement in validation loss

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_jaccard = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                frame, true_masks = batch
                bs, C, H, W = frame.shape
                
                frame = frame.to(device, dtype=torch.float32)
                
                # reshape true_masks from (bs, 1, H, W) to (bs, H, W)
                true_masks = true_masks.squeeze(1)
                true_masks = true_masks.to(device,  dtype=torch.long)

                masks_pred = model(frame)

                # print(f'masks_pred shape: {masks_pred.shape}')
                # print(masks_pred)

                loss = criterion(masks_pred, true_masks)

                # Apply softmax to masks_pred to get probabilities
                masks_pred_softmax = F.softmax(masks_pred, dim=1)

                # print(f'masks_pred_softmax shape: {masks_pred_softmax.shape}')
                # print(masks_pred_softmax)
                
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

                pbar.update(frame.shape[0])
                epoch_loss += loss.item()

                experiment.log({"loss": loss.item()})

                pbar.set_postfix(**{'loss (batch)': loss.item()})


        # Validation part after each epoch
        val_jaccard, val_loss = evaluate(model, val_loader, device, amp)
        scheduler.step(val_jaccard)

        # Log metrics to Weights & Biases
        average_jaccard = epoch_jaccard / len(train_loader)
        experiment.log({"Epoch": epoch, "Train Loss": epoch_loss / len(train_loader), "Train Jaccard": average_jaccard, "Val Loss": val_loss, "Cal Jaccard": val_jaccard})
        logging.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss / len(train_loader)}, Train Jaccard: {average_jaccard}, Val Loss: {val_loss}, Val Jaccard: {val_jaccard}')
        
        if val_jaccard> best_jac:
            epochs_no_improve = 0 # Reset patience

            # If current model is better, delete the previous best model checkpoint
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)

            # Save the current model checkpoint
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            best_jac = val_jaccard
            best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'New best model saved at epoch {epoch + 1} with Jaccard score: {best_jac}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                logging.info(f'Early stopping at epoch {epoch + 1}')
                break

    print("Training completed")
    print(f'Best Jaccard score: {best_jac}')
    if best_model_path:
        print(f'Best model saved at {best_model_path}')
    experiment.finish()

@torch.inference_mode()
def evaluate(model, dataloader, device, amp):
    model.eval()
    num_val_batches = len(dataloader)
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=model.n_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    total_jaccard = 0
    total_loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            frames, true_mask = batch
            bs, C, H, W = frames.shape

            # conver to (bs, seq_len * C, H, W)
            frames = frames.to(device, dtype=torch.float32, memory_format=torch.channels_last)

            true_mask = true_mask.squeeze(1)
            true_mask = true_mask.to(device,  dtype=torch.long)
            
            masks_pred = model(frames)
            masks_pred_softmax = F.softmax(masks_pred, dim=1)
            mask_pred_argmax = torch.argmax(masks_pred_softmax, dim=1)
            jac_score = jaccard(mask_pred_argmax, true_mask).item()

            # Calculate Jaccard Index for each batch and accumulate
            batch_jaccard = jac_score
            total_jaccard += batch_jaccard

            # Calculate Cross-Entropy Loss
            loss = criterion(masks_pred, true_mask)
            total_loss += loss.item()

    model.train()
    
    avg_jaccard = total_jaccard / max(num_val_batches, 1)
    avg_loss = total_loss / max(num_val_batches, 1)

    return avg_jaccard, avg_loss

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

    # Dimension of the input to the model
    input_channel = 3  # 3 RGB channels per frame, 11 frames
    output_channel = 49  # 22 classes for segmentation

    # Initialize the UNet model
    model = UNet(input_channel, output_channel)
    model.to(device)

    train_model(
        model=model,
        device=device,
        train_transform=train_transform,
        val_transform=val_transform,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        gradient_clipping=args.gradient_clipping,
        amp=args.amp,
        model_name=args.model_name,
        subset_test=args.subset_test,
        root_dir=args.root_dir
    )


   





