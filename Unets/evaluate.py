import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchmetrics

from utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(model, dataloader, device, amp):
    model.eval()
    num_val_batches = len(dataloader)
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=model.n_classes).to(device)

    total_jaccard = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            frames, mask_true = batch
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

            # Calculate Jaccard Index for each batch and accumulate
            batch_jaccard = jaccard(jac_score, mask_true).item()
            total_jaccard += batch_jaccard

    model.train()
    # Return the average Jaccard Index
    return total_jaccard / max(num_val_batches, 1)
