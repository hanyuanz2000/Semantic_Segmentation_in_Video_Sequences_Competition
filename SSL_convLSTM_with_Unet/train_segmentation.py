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
import numpy as np

# Your custom modules
from models.unet_model import UNet
from models.Reconstructor_mini import VideoFrameReconstructor_Mini
from models.Reconstructor import VideoFrameReconstructor
from utils import data_loading
from utils import customized_transform
from basic_config import root_dir
from utils.dice_score import dice_loss
from evaluate_segmentation import evaluate
from datetime import datetime

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# export PYTORCH_ENABLE_MPS_FALLBACK=1

# Function to parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--amp', type=bool, default=False, help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--model_name', type=str, default='Unet_Pred_on_SSL_Predited_Frame', help='Name of the model')
    parser.add_argument('--path_to_reconstructor_weights', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/SSL_convLSTM_with_Unet/checkpoints/Reconstructor Mini_Dec05-10:00_Opti_RMSprop_LR_0.001_BS_4_WD_1e-06_Mom_0.9_GradClip_1.0/best_model_epoch_1.pth', help='Path to the weights of the reconstructor')
    parser.add_argument('--reconstructor_name', type=str, default='VideoFrameReconstructor_Mini', help='Name of the reconstructor')
    parser.add_argument('--subset_test', type=bool, default=False, help='Whether to use a subset of the data for testing')

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
        reconstructor_name,
        reconstructor,
        subset_test
    ):

    # set project name and initialize experiment 
    project_name = model_name + 'with' + reconstructor_name
    
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
    train_set = data_loading.Labeled_Segementation_Dataset(root_dir=root_dir, subset='train', transform=train_transform)
    val_set = data_loading.Labeled_Segementation_Dataset(root_dir=root_dir, subset='val', transform=val_transform)
    n_train, n_val = len(train_set), len(val_set)

    if subset_test: # Only use a subset of the data for testing
        # Define the size of the subset as 5% of the dataset
        train_subset_size = int(0.05 * n_train)
        val_subset_size = int(0.05 * n_val)

        # Generate random indices for train and validation subsets
        train_subset_indices = np.random.choice(range(n_train), train_subset_size, replace=False)
        val_subset_indices = np.random.choice(range(n_val), val_subset_size, replace=False)

        # Create subsets
        train_subset = Subset(train_set, train_subset_indices)
        val_subset = Subset(val_set, val_subset_indices)
    
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    
    else: # Use the entire dataset
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

    logging.info(f'Initializing training on {len(train_set)} training images and {len(val_set)} validation images')
    logging.info(f'Using device {device}')
    logging.info(f'Image size: {train_set[0][0][0].size()}')

    reconstructor.eval()  # Set to evaluation mode

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
                frame = frame.to(device, dtype=torch.float32)
                
                # reshape true_masks from (bs, 1, H, W) to (bs, H, W)
                true_masks = true_masks.squeeze(1)
                true_masks = true_masks.to(device, dtype=torch.long)

                # Predict the 22nd frame using the reconstructor
                with torch.no_grad():  # No gradients needed for the reconstructor
                    predicted_frames = reconstructor(frame)
                    predicted_frames.to(device, dtype=torch.float32)
            
                # Predict the segmentation masks using U-Net
                masks_pred = model(predicted_frames)

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

                pbar.update(frame.shape[0])
                epoch_loss += loss.item()

                experiment.log({"loss": loss.item()})

                pbar.set_postfix(**{'loss (batch)': loss.item()})
            

        # Validation part after each epoch
        val_jaccard = evaluate(model, reconstructor, val_loader, device, amp)
        scheduler.step(val_jaccard)

        # Log metrics to Weights & Biases
        average_jaccard = epoch_jaccard / len(train_loader)
        experiment.log({"Epoch": epoch, "Train Loss": epoch_loss / len(train_loader), "Train Jaccard": average_jaccard, "Val Jaccard": val_jaccard})
        logging.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss / len(train_loader)}, Train Jaccard: {average_jaccard}, Val Jaccard: {val_jaccard}')
        
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
    output_channel = 49

    # Initialize the UNet model
    model = UNet(input_channel, output_channel)
    model.to(device)

    # Load the pre-trained reconstructor
    if args.reconstructor_name == 'VideoFrameReconstructor_Mini':
        reconstructor = VideoFrameReconstructor_Mini(C=3, num_frames=11, LSTM_hidden_size=256, num_layers=2)
    elif args.reconstructor_name == 'VideoFrameReconstructor':
        reconstructor = VideoFrameReconstructor(C=3, num_frames=11, LSTM_hidden_size=256, num_layers=2)
    reconstructor.to(device)
    reconstructor.load_state_dict(torch.load(args.path_to_reconstructor_weights, map_location=device))

    args = get_args()

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
        reconstructor_name = args.reconstructor_name,
        reconstructor = reconstructor,
        subset_test = args.subset_test
    )


   





