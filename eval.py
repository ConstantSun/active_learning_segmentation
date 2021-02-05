import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from matplotlib import pyplot as plt
from dice_loss import dice_coeff, iou_numpy, iou_pytorch

global GAUSS_ITERATION
GAUSS_ITERATION = 30


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
            # with torch.no_grad():
                # logging.info(f"EVAL - img shape: {imgs.shape}")
                # print(f"EVAL - img shape: {imgs.shape}")
                # mask_pred = net(imgs)

            y_pred_samples = []
            for i in range(GAUSS_ITERATION):
                with torch.no_grad():
                    logits = net(imgs)
                    y_pred = torch.sigmoid(logits)
                    # y_pred = (y_pred > 0.5).float()
                    y_pred = y_pred[:, :1, :, :]
                    y_pred_samples.append(y_pred[:, 0, :, :])  # y_pred_samples's shape: (inx, bat, H, W )
            y_pred_samples = torch.stack(y_pred_samples, dim=0)
            y_pred_samples = y_pred_samples.type(torch.cuda.FloatTensor)
            mean_y_pred = y_pred_samples.mean(dim=0)  # shape: batch, H, W
            pred = mean_y_pred.unsqueeze(1)

            if n_classes > 1:
                tot += F.cross_entropy(pred, true_masks).item()
            else:
                pred = (pred > 0.5).type(torch.cuda.FloatTensor)
                pred = pred[:, :1, :, :]

                # print(f"*******\npred, true_masks shape: {pred.shape}, {true_masks.shape}")
                tot += dice_coeff(pred, true_masks, device).item()
                iou += iou_pytorch(pred, true_masks).item()
            pbar.update()
    net.train()
    return tot / n_val, iou / n_val