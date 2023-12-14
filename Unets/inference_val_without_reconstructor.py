"""
This script is used to do inference on the validation dataset and test the inference result.
We will directly use the 11th frame to do inference (Pass the 11th frame to the segmentation model).
We do this due to the inadquecy the reconstruction quality of the Reconstructor model.
This is different from the inference_hidden_with_reconstuctor.py
where we use the Reconstructor model to predict the 22nd frame and then pass the 22nd frame to the segmentation model.
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
    parser.add_argument('--saved_recon_model_dir', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/SSL_convLSTM_with_Unet/checkpoints/recons_best_model_07.pth', help='Directory to save the trained model')
    parser.add_argument('--LSTM_hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in LSTM')

    return parser.parse_args()

def inferece(
        seg_model,
        device,
        root_dir,
    ):
    # Initialize the dataset
    validation_set = data_loading.Labeled_Segementation_Dataset(root_dir=root_dir, subset='val', transform=customized_transform.SegmentationValidationTransform())

    # Initialize the dataloader
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=True)

    seg_model.eval()

    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=seg_model.n_classes).to(device)
    jaccard_record = []
    i = 0

    with torch.no_grad():
        for frames, mask in tqdm(val_loader, desc="Inference"):
            bs, seq_len, C, H, W = frames.shape
            last_frame = frames[:, -1, :, :, :]
            last_frame = last_frame.to(device, dtype=torch.float32)
            mask = mask.to(device)
            
            masks_pred = seg_model(last_frame)

            masks_pred_softmax = F.softmax(masks_pred, dim=1)
            mask_pred_argmax = torch.argmax(masks_pred_softmax, dim=1)

            # print some info for the first batch
            if i == 0:
                # print shape
                print(f'mask shape: {mask.shape}')
                print(f'frames shape: {frames.shape}')
                print(f'11th frames shape: {last_frame.shape}')
                print(f'masks_pred shape: {masks_pred.shape}')
                print(f'mask_pred_argmax shape: {mask_pred_argmax.shape}')

                # also get the last frame from frames
                last_frame = frames[0, -1, :, :, :]
                print(f'last frame shape: {last_frame.shape}')

                # visualize the mask and the predicted mask
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
                axes[0].imshow(mask.squeeze(0).squeeze(0).cpu().numpy())
                axes[0].set_title('Ground Truth')
                plot_frame(axes[1], last_frame, '11th Frame')
                axes[2].imshow(mask_pred_argmax.squeeze(0).cpu().numpy())
                axes[2].set_title('Predicted Mask')

            plt.show()

            # calculate jaccard score
            jaccard_score = jaccard(mask_pred_argmax.squeeze(0), mask.squeeze(0).squeeze(0))
            jaccard_record.append(jaccard_score)

            i += 1
            if i % 100 == 0:
                print(f'Processed {i} batches')
        
        jaccard_scores_numpy = [score.cpu().numpy() for score in jaccard_record]
        # Calculate mean using numpy
        mean_jaccard_score = np.mean(jaccard_scores_numpy)
        print(f'Mean Jaccard score: {mean_jaccard_score}')

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

    # Inference
    inferece(seg_model, device, args.root_dir)