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
    parser.add_argument('--saved_model_dir', type=str, default='/Users/zhanghanyuan/Document/Git/Semantic_Segmentation_in_Video_Sequences_Competition/Unets/checkpoints/Unet_Direct_Pred_Dec06-15:27_lr0.001_bs4_wd1e-06_mom0.9_gc1.0/best_model_epoch_1.pth', help='Directory to save the trained model')

    return parser.parse_args()

def inferece(
        model,
        device,
        root_dir,
    ):
    # Initialize the dataset
    hidden_dataset = data_loading.Hidden_Dataset(
        root_dir=root_dir,
    )

    # Initialize the dataloader
    hidden_loader = DataLoader(hidden_dataset, batch_size=1, shuffle=False)

    model.eval()
    output_tensors = []

    with torch.no_grad():
        for frames, _ in tqdm(hidden_loader, desc="Inference"):
            bs, seq_len, C, H, W = frames.shape
                
            # conver to (bs, seq_len * C, H, W)
            frames = frames.contiguous()
            frames= frames.view(bs, seq_len*C, H, W)
            frames = frames.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            
            masks_pred = model(frames)
            masks_pred_softmax = F.softmax(masks_pred, dim=1)
            mask_pred_argmax = torch.argmax(masks_pred_softmax, dim=1)

            # mask_pred_argmax has shape (1, 160, 240)
            output_tensors.append(mask_pred_argmax.cpu())

    # output tensor is a list 2000 tensors, each has shape (1, 160, 240), I want to concatenate them to a tensor with shape (2000, 160, 240)
    result_tensor = torch.cat(output_tensors, dim=0)
    # check the shape of the result tensor
    print(f'Result tensor shape: {result_tensor.shape}') # (2000, 160, 240)
    
    return result_tensor

def test_inference(
        model,
        device,
        root_dir,
    ):
    '''
    test with validation dataset
    check if the n*H*W tensor we generate is correct
    '''
    # Initialize the dataset
    val_set = data_loading.Labeled_Segementation_Dataset(root_dir=root_dir, subset='val', transform=customized_transform.SegmentationValidationTransform())
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model.eval()
    jaccard_result = [] # List to store Jaccard index for each sample
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=model.n_classes).to(device)

    with torch.no_grad():
        for frames, true_mask in tqdm(val_loader, desc="Inference"):
            bs, seq_len, C, H, W = frames.shape
                
            # conver to (bs, seq_len * C, H, W)
            frames = frames.contiguous()
            frames= frames.view(bs, seq_len*C, H, W)
            frames = frames.to(device, dtype=torch.float32, memory_format=torch.channels_last)

            true_mask = true_mask.squeeze(1)
            true_mask = true_mask.to(device,  dtype=torch.long)
            
            masks_pred = model(frames)
            masks_pred_softmax = F.softmax(masks_pred, dim=1)
            mask_pred_argmax = torch.argmax(masks_pred_softmax, dim=1)
            jac_score = jaccard(mask_pred_argmax, true_mask).item()

            # calculate Jaccard index for each sample
            jaccard_result.append(jac_score)
    
    avg_jaccard = np.mean(jaccard_result)
    return avg_jaccard


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
    input_channel = 33  # 3 RGB channels per frame, 11 frames
    output_channel = 49  # 22 classes for segmentation

    # Initialize the UNet model
    model = UNet(input_channel, output_channel)
    model.to(device)

    # Load the trained model
    model.load_state_dict(torch.load(args.saved_model_dir))

    # Inference on hidden dataset
    inference_result = inferece(
        model,
        device,
        args.root_dir,
    )

    # Save the inference result
    inference_result = inference_result.numpy()
    
    # check the shape of the inference result again
    print(f'Inference result shape: {inference_result.shape}') # (2000, 160, 240)
    np.save('inference_result.npy', inference_result)

    # # Test the inference result
    # avg_jaccard = test_inference(
    #     model,
    #     device,
    #     args.root_dir,
    # )
    # print(f'Average Jaccard index: {avg_jaccard}')