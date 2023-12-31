"""
This script is used to do inference on the hidden dataset and save the inference result.
The data of the first 11 frames are fed into the Reconstructor model to predict the 22nd frame.
The predicted 22nd frame is then fed into the segmentation model to predict the mask.
"""
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
from models.Reconstructor_LessDownSample import VideoFrameReconstructor_LessDownSample
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
    parser.add_argument('--saved_seg_model_dir', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Unets/checkpoints/Unets_best_model_epoch_11.pth', help='Directory to save the trained model')
    parser.add_argument('--saved_recon_model_dir', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/SSL_convLSTM_Reconstructor/checkpoints/recon_best_model_71.pth', help='Directory to save the trained model')
    parser.add_argument('--LSTM_hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in LSTM')

    return parser.parse_args()


def inferece(
        seg_model,
        recons_model,
        device,
        root_dir,
    ):
    # Initialize the dataset
    hidden_dataset = data_loading.Hidden_Dataset(
        root_dir=root_dir,
        subset='hidden',
        transform=customized_transform.SegmentationValidationTransform()
    )

    # Initialize the dataloader
    hidden_loader = DataLoader(hidden_dataset, batch_size=1, shuffle=False)

    seg_model.eval()
    recons_model.eval()

    tensor_record = []
    i = 0

    with torch.no_grad():
        for frames, _, video_name in tqdm(hidden_loader, desc="Inference"):
            bs, seq_len, C, H, W = frames.shape
            
            # conver to (bs, seq_len * C, H, W)
            frames = frames.to(device, dtype=torch.float32)
            
            predicted_22nd_frame = recons_model(frames)
            predicted_22nd_frame = predicted_22nd_frame.to(device)
            
            masks_pred = seg_model(predicted_22nd_frame) # [1, 49, 160, 240]
            masks_pred_softmax = F.softmax(masks_pred, dim=1) # [1, 49, 160, 240]
            mask_pred_argmax = torch.argmax(masks_pred_softmax, dim=1) # [1, 160, 240]

            if i == 0:
                # print shape
                print(f'frames shape: {frames.shape}')
                print(f'predicted_22nd_frame shape: {predicted_22nd_frame.shape}')
                print(f'masks_pred shape: {masks_pred.shape}')
                print(f'masks_pred_softmax shape: {masks_pred_softmax.shape}')
                print(f'mask_pred_argmax shape: {mask_pred_argmax.shape}')

                # visualize the mask and the predicted mask
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
                plot_frame(axes[0], frames[0, -1, :, :, :], '11th Frame')
                axes[1].imshow(mask_pred_argmax.squeeze(0).cpu().numpy())
                plt.show()

                first_pred_mask = mask_pred_argmax.squeeze(0).cpu().numpy()
            
            if i <= 10:
                 print(f'video_name: {video_name}')

            if i == 10:
                break

            # mask_pred_argmax has shape (1, 160, 240)
            tensor_record.append(mask_pred_argmax.cpu())

            i += 1

    # output tensor is a list 2000 tensors, each has shape (1, 160, 240), I want to concatenate them to a tensor with shape (2000, 160, 240)
    result_tensor = torch.cat(tensor_record, dim=0)
    # check the shape of the result tensor
    print(f'Result tensor shape: {result_tensor.shape}') # (2000, 160, 240)

    assert result_tensor.shape == (2000, 160, 240), 'Result tensor shape is not correct'
    assert np.array_equal(first_pred_mask, result_tensor[0].squeeze(0).cpu().numpy()), 'First predicted mask is not correct'
    
    return result_tensor

def plot_frame(ax, frame, title):
    """ Plot a single frame with title. """
    frame_image = frame.cpu().numpy()
    frame_image = np.transpose(frame_image, (1, 2, 0))

    # Normalize from [-1, 1] to [0, 1]
    frame_image = (frame_image + 1) / 2
    frame_image = np.clip(frame_image, 0, 1)  # Ensuring the values are within [0, 1]

    ax.imshow(frame_image)
    ax.set_title(title)

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

    # Initialize the UNet model
    seg_model = UNet(3, 49)
    seg_model.to(device)
    seg_model.load_state_dict(torch.load(args.saved_seg_model_dir, map_location=device))

    # Initialize the Reconstructor model
    recons_model = VideoFrameReconstructor_LessDownSample(C=3, num_frames=11, LSTM_hidden_size=args.LSTM_hidden_size, num_layers = args.num_layers)
    recons_model.to(device)
    recons_model.load_state_dict(torch.load(args.saved_recon_model_dir, map_location=device))

    # Inference
    inference_result = inferece(seg_model, recons_model, device, args.root_dir)
    inference_result = inference_result.numpy()
    print(f'Inference result shape: {inference_result.shape}') # (2000, 160, 240)
    # np.save('inference_result.npy', inference_result)