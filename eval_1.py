import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from matplotlib import pyplot as plt
from dice_loss import dice_coeff, iou_numpy, iou_pytorch


def eval_net(net, loader, n_classes, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    iou = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type) # BHWC
            true_masks = true_masks[:, :1, :, :]


            with torch.no_grad():
                # logging.info(f"EVAL - img shape: {imgs.shape}")
                # print(f"EVAL - img shape: {imgs.shape}")
                mask_pred = net(imgs)

            if n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                pred = pred[:, :1, :, :]

                # print(f"*******\npred, true_masks shape: {pred.shape}, {true_masks.shape}")
                tot += dice_coeff(pred, true_masks, device).item()
                iou += iou_pytorch(pred, true_masks).item()

            pbar.update()

    net.train()
    return tot / n_val, iou / n_val