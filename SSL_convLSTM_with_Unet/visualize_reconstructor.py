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
import cv2
from utils import data_loading
import matplotlib.pyplot as plt
import numpy as np
from utils import customized_transform

from models.Reconstructor_LessDownSample import VideoFrameReconstructor_LessDownSample
from models.ReconstructorMini import VideoFrameReconstructor_Mini
from models.Reconstructor import VideoFrameReconstructor

from torchvision.transforms import functional as F

# Function to parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Inference on hidden dataset')
    parser.add_argument('--root_dir', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Data', help='Root directory of the dataset')
    parser.add_argument('--saved_model_dir', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/SSL_convLSTM_with_Unet/checkpoints/best_model_epoch_1.pth', help='Directory to save the trained model')
    parser.add_argument('--model_name', type=str, default='Reconstructor Less DownSample', help='Choose the model to train')
    parser.add_argument('--LSTM_hidden_size', type=int, default=512, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in LSTM')

    return parser.parse_args()


def visualize(
        model,
        device,
        root_dir,
    ):
    # Initialize the dataset
    train_dataset = data_loading.SSL_Reconstruction_Dataset(root_dir=root_dir, subset='train', transform=customized_transform.SegmentationValidationTransform())
    val_dataset = data_loading.SSL_Reconstruction_Dataset(root_dir=root_dir, subset='val', transform=customized_transform.SegmentationValidationTransform())

    # Initialize the dataloader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model.eval()
    train_last_4_frames = []
    val_last_4_frames = []

    # visualize on 2 samples from train dataset
    with torch.no_grad():
        for frames, future_frame in tqdm(train_dataset, desc="Inference"):
            seq_len, C, H, W = frames.shape
            # conver to (bs, seq_len,  C, H, W)
            frames = frames.contiguous()
            frames= frames.view(1, seq_len, C, H, W)
            print(f'frames shape: {frames.shape}')

            # frames has shape [B, T, C, H, W]
            frames = frames.to(device, dtype=torch.float32)
            future_frame = future_frame.to(device, dtype=torch.float32)

            # future frame print
            print(f'future_frame shape: {future_frame.shape}')
            print(future_frame)

            # Get the predicted frame from the model
            predicted_future_frame = model(frames)

            print(f'predicted_frame shape: {predicted_future_frame.shape}')
            print(predicted_future_frame)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

            normalize_true_frame = F.normalize(future_frame.squeeze(0), mean=mean, std=std)

            # print the normalized true frame
            print(f'normalize_true_frame shape: {normalize_true_frame.shape}')
            print(normalize_true_frame)

            normalize_predicted_frame = (predicted_future_frame - mean) / std
            
            # get (c, h, w) from (bs, T, C, H, W) for each sample
            frames = frames.squeeze(0) # # Now the shape is [T, C, H, W]
            for t in range(7, frames.size(0)): # get the last 4 frames
                slice = frames[t, :, :, :]
                train_last_4_frames.append(slice)
            break
    
    # Set up the subplot grid
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(12, 6))  # Adjust figsize as needed

    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()
    # Plot each image
    for i, slice in enumerate(train_last_4_frames):
        image = slice.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        axes_flat[i].imshow(image)

    # plot the true frame
    image = normalize_true_frame.squeeze(0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    axes_flat[4].imshow(image)

    # plot the predicted frame
    image = normalize_predicted_frame.squeeze(0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    axes_flat[5].imshow(image)
    
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

    # Initialize the model
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


    # Load the state dict of the trained model
    model.load_state_dict(torch.load(args.saved_model_dir, map_location=device))

    # Inference and visualize
    visualize(
        model,
        device,
        args.root_dir,

    )


