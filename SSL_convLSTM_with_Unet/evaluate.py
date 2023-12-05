import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchmetrics

from utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=net.n_classes).to(device)

    total_jaccard = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch
            # conver to (bs, seq_len * C, H, W)
            bs, seq_len, C, H, W = image.shape
            image = image.contiguous()
            image= image.view(bs, seq_len*C, H, W)
            image = image.to(device, dtype=torch.float32, memory_format=torch.channels_last)
            # move images and labels to correct device and type
            if mask_true.ndim > 3:  # This means masks are one-hot encoded
                mask_true = torch.argmax(mask_true, dim=1)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            # Apply softmax to mask_pred to get probabilities
            mask_pred_softmax = F.softmax(mask_pred, dim=1)
            # Convert mask_pred to one-hot format
            mask_pred_one_hot = F.one_hot(mask_pred.argmax(dim=1), num_classes=net.n_classes)
            mask_pred_one_hot = mask_pred_one_hot.permute(0, 3, 1, 2).float()

            # Calculate Jaccard Index for each batch and accumulate
            batch_jaccard = jaccard(mask_pred_one_hot, mask_true).item()
            total_jaccard += batch_jaccard

    net.train()
    # Return the average Jaccard Index
    return total_jaccard / max(num_val_batches, 1)
