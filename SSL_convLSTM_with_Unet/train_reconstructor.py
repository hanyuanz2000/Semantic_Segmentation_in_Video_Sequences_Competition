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
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio 
import numpy as np

# Your custom modules
from models.ReconstructorMini import VideoFrameReconstructor_Mini
from models.Reconstructor import VideoFrameReconstructor
from models.Reconstructor_LessDownSample import VideoFrameReconstructor_LessDownSample
from utils import data_loading
from utils import customized_transform
from utils.dice_score import dice_loss
from evaluate_reconstructor import evaluate
from datetime import datetime

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# export PYTORCH_ENABLE_MPS_FALLBACK=1

# Function to parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the Reconstructor model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--amp', type=bool, default=False, help='Enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--model_name', type=str, default='Reconstructor Mini', help='Choose the model to train')
    parser.add_argument('--optimizer', type=str, default='RMSprop', help='Choose the optimizer')
    parser.add_argument('--subset_test', type=bool, default=False, help='Whether to use a subset of the data for testing'),
    parser.add_argument('--root_dir', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data', help='Root directory of the dataset')
    parser.add_argument('--LSTM_hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in LSTM')
    
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
        optimizer_choice,
        subset_test,
        root_dir,
        lstm_hidden_size,
        num_layers
    ):

    # set project name and initialize experiment 
    project_name = f'{model_name}_training'
    
    now = datetime.now()
    dt_string = 'Dec' + now.strftime("%d") + '-' + now.strftime("%H:%M")
    config_string = f'lstmhiddensize_{lstm_hidden_size}_nlayers_{num_layers}_Opti_{optimizer_choice}_LR_{learning_rate}_BS_{batch_size}_WD_{weight_decay}_Mom_{momentum}_GradClip_{gradient_clipping}'
    experiment_name = f'{dt_string}_{config_string}'
    logging.info(f'Experiment name: {experiment_name}')

    # Create a unique directory for this experiment204
    unique_dir = f'{model_name}_{dt_string}_{config_string}'
    checkpoint_dir = os.path.join('./checkpoints', unique_dir)
    
    # Initialize Weights & Biases logging
    experiment = wandb.init(project=project_name, name=experiment_name, config={
        "model_name": model_name,
        "optimizer": optimizer_choice,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "gradient_clipping": gradient_clipping,
    })
    
    # load data
    # for this SSL, we use unlabeled data for training and labeled data for validation, so that in the second phase, model can't cheat by using labeled data
    train_set = data_loading.SSL_Reconstruction_Dataset(root_dir=root_dir, subset='train', transform=train_transform)
    val_set = data_loading.SSL_Reconstruction_Dataset(root_dir=root_dir, subset='val', transform=val_transform)
    n_train, n_val = len(train_set), len(val_set)

    if subset_test: # Only use a subset of the data for testing
        # Define the size of the subset as 5% of the dataset
        train_subset_size = int(0.2 * n_train)
        val_subset_size = int(0.2 * n_val)

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

    logging.info(f'Initializing training on {len(train_set)} training data and {len(val_set)} validation images')
    logging.info(f'Using device {device}')
    logging.info(f'Training tensor shape: {train_set[0][0].shape}')

    # Initialize optimizer, loss function and learning rate scheduler
    # Initialize optimizer based on choice
    if optimizer_choice == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_choice == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    criterion = nn.MSELoss()

    # Initialize metrics
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    best_metrics = {
        'mse_loss': float('inf'),
        'psnr': 0.0,
        'ssim': 0.0
    }
    best_model_path = None

    patience = 10 # Number of epochs to wait before reducing learning rate
    epochs_no_improve = 0 # Number of epochs with no improvement in validation loss

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                frames, future_frame = batch
                # frames has shape [B, T, C, H, W]
                frames = frames.to(device, dtype=torch.float32)
                future_frame = future_frame.to(device, dtype=torch.float32)

                predicted_frame = model(frames)

                mse_loss = criterion(predicted_frame, future_frame)

                optimizer.zero_grad()
                mse_loss.backward()
                optimizer.step()

                pbar.update(frames.shape[0])
                epoch_loss += mse_loss.item()

                psnr_score = psnr(predicted_frame, future_frame).item()
                ssim_score = ssim(predicted_frame, future_frame).item()
                epoch_psnr += psnr_score
                epoch_ssim += ssim_score

                experiment.log({
                    "Train Loss": mse_loss.item(),
                    "Train PSNR": psnr_score,
                    "Train SSIM": ssim_score,
                })
                pbar.set_postfix(**{'loss (batch)': mse_loss.item()})
        
        # Calculate average metrics for the epoch (train)
        average_psnr = epoch_psnr / len(train_loader)
        average_ssim = epoch_ssim / len(train_loader)

        # Validation part after each epoch
        avg_mse_loss, avg_psnr, avg_ssim = evaluate(model, val_loader, device)  # Updated evaluate function call
        scheduler.step(avg_mse_loss)  # Updated scheduler to use mse_loss

        # Log metrics to Weights & Biases
        experiment.log({
            "Epoch": epoch,
            "Train MSE Loss": epoch_loss / len(train_loader),
            "Train PSNR": average_psnr,
            "Train SSIM": average_ssim,
            "Val MSE Loss": avg_mse_loss,
            "Val PSNR": avg_psnr,
            "Val SSIM": avg_ssim
        })
        

        # Determine if the model should be saved based on multiple metrics improving
        improved_metrics = 0
        if avg_mse_loss < best_metrics['mse_loss']:
            best_metrics['mse_loss'] = avg_mse_loss
            improved_metrics += 1
        if avg_psnr > best_metrics['psnr']:
            best_metrics['psnr'] = avg_psnr
            improved_metrics += 1
        if avg_ssim > best_metrics['ssim']:
            best_metrics['ssim'] = avg_ssim
            improved_metrics += 1

        # Save the model if at least two metrics improved
        if improved_metrics >= 2:
            # Reset patience
            epochs_no_improve = 0

            # If current model is better, delete the previous best model checkpoint
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            # Save the current model checkpoint
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'New best model saved at epoch {epoch + 1} with MSE loss: {avg_mse_loss}, PSNR: {avg_psnr}, SSIM: {avg_ssim}')
        
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                logging.info(f'Early stopping at epoch {epoch + 1}')
                break

    logging.info(f'Finished training after {epoch + 1} epochs')
    logging.info(f'Best metrics: MSE loss: {best_metrics["mse_loss"]}, PSNR: {best_metrics["psnr"]}, SSIM: {best_metrics["ssim"]}')
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

    # choose model
    if args.model_name == 'Reconstructor':
        # Initialize the model
        model = VideoFrameReconstructor(C=3, num_frames=11, LSTM_hidden_size=args.LSTM_hidden_size, num_layers = args.num_layers)
        model.to(device)
        print(f'Number of parameters in the model: {sum(p.numel() for p in model.parameters())}')
    elif args.model_name == 'Reconstructor Mini':
        # Initialize the model
        model = VideoFrameReconstructor_Mini(C=3, num_frames=11, LSTM_hidden_size=args.LSTM_hidden_size, num_layers = args.num_layers)
        model.to(device)
        print(f'Number of parameters in the model: {sum(p.numel() for p in model.parameters())}')
    elif args.model_name == 'Reconstructor Less DownSample':
        # Initialize the model
        model = VideoFrameReconstructor_LessDownSample(C=3, num_frames=11, LSTM_hidden_size=args.LSTM_hidden_size, num_layers = args.num_layers)
        model.to(device)
        print(f'Number of parameters in the model: {sum(p.numel() for p in model.parameters())}')

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
        optimizer_choice=args.optimizer,
        subset_test=args.subset_test,
        root_dir=args.root_dir,
        lstm_hidden_size=args.LSTM_hidden_size,
        num_layers=args.num_layers
    )


   