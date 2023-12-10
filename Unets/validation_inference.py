import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt

# custom modules
from models.unet_model import UNet
from utils import data_loading
from utils import customized_transform
from utils.dice_score import dice_loss
from evaluate import evaluate
from datetime import datetime
import numpy as np

# Function to parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Inference on hidden dataset')
    parser.add_argument('--root_dir', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data', help='Root directory of the dataset')
    parser.add_argument('--saved_model_dir', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Unets/checkpoints/Unet_Direct_Pred_Dec09-21:41_lr0.001_bs2_wd0.0001_mom0.9_gc1.0/best_model_epoch_1.pth', help='Directory to save the trained model')

    return parser.parse_args()

def inferece(
        model,
        device,
        root_dir,
    ):
    # Initialize the dataset
    validation_set = data_loading.LastFrame_and_Mask_Dataset(root_dir=root_dir, subset='val', transform=customized_transform.SegmentationValidationTransform())

    # Initialize the dataloader
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False)

    model.eval()

    with torch.no_grad():
        for frame, mask in tqdm(val_loader, desc="Inference"):
            bs, C, H, W = frame.shape
                
            # conver to (bs, seq_len * C, H, W)
            frame = frame.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            
            masks_pred = model(frame)
            masks_pred_softmax = F.softmax(masks_pred, dim=1)
            mask_pred_argmax = torch.argmax(masks_pred_softmax, dim=1)

            print(f'mask_pred_argmax shape: {mask_pred_argmax.shape}')

            # visualize the mask and the predicted mask
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            axes[0].imshow(mask.squeeze(0).squeeze(0))
            axes[1].imshow(mask_pred_argmax.squeeze(0).cpu().numpy())
            plt.show()
    

if __name__ == '__main__':
    args = get_args()
    logging.info(args)

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

    # Dimension of the input to the model
    input_channel = 3  # 3 RGB channels per frame, 11 frames
    output_channel = 49  # 22 classes for segmentation

    # Initialize the UNet model
    model = UNet(input_channel, output_channel)
    model.to(device)

    # Load the trained model
    model.load_state_dict(torch.load(args.saved_model_dir))

    # Inference on hidden dataset
    inferece = inferece(
        model,
        device,
        args.root_dir,
    )