import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchmetrics

@torch.inference_mode()
def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)

    # Initialize metrics
    mse_metric = torchmetrics.MeanSquaredError().to(device)
    psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_mse = 0
    total_psnr = 0
    total_ssim = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        frames, future_frame_true = batch
        frames = frames.to(device, dtype=torch.float32)
        future_frame_true = future_frame_true.to(device, dtype=torch.float32)

        # predict the future frame
        future_frame_pred = net(frames)

        # Update metrics
        total_mse += mse_metric(future_frame_pred, future_frame_true).item()
        total_psnr += psnr_metric(future_frame_pred, future_frame_true).item()
        total_ssim += ssim_metric(future_frame_pred, future_frame_true).item()

    net.train()

    # Return the average metrics
    avg_mse = total_mse / num_val_batches
    avg_psnr = total_psnr / num_val_batches
    avg_ssim = total_ssim / num_val_batches

    return avg_mse, avg_psnr, avg_ssim
