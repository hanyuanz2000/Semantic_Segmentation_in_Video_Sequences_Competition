import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchmetrics


@torch.inference_mode()
def evaluate(net, reconstructor, dataloader, device, amp):
    net.eval()  # Ensure the network is in evaluation mode
    reconstructor.eval()  # Ensure the reconstructor is in evaluation mode
    num_val_batches = len(dataloader)
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=net.n_classes).to(device)

    total_jaccard = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            frames, true_masks = batch

            frames = frames.to(device, dtype=torch.float32)
            
            true_masks = true_masks.squeeze(1)
            true_masks = true_masks.to(device, dtype=torch.long)

            # Reconstruct the frames using the reconstructor
            with torch.no_grad():
                reconstructed_frames = reconstructor(frames)
                reconstructed_frames.to(device, dtype=torch.float32)

            # Predict the segmentation masks using the main network
            masks_pred = net(reconstructed_frames)

            # Apply softmax to masks_pred to get probabilities
            masks_pred_softmax = F.softmax(masks_pred, dim=1)

            # Calculate Jaccard Index for each batch and accumulate
            jaccard_score = jaccard(masks_pred_softmax, F.one_hot(true_masks, num_classes=net.n_classes).permute(0, 3, 1, 2).float())
            total_jaccard += jaccard_score.item()

    # Return the average Jaccard Index
    return total_jaccard / num_val_batches
