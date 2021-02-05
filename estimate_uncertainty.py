import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from matplotlib import pyplot as plt
from ece_metric import *
global GAUSS_ITERATION
GAUSS_ITERATION = 100


def visualize_ece_of_training_images_data(net, loader, n_classes, device):
    """
    Visualize Expected calibration error of all images in training data set
    """
    net.eval()
    mask_type = torch.float32 if n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    ece_values = []
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type) # BHWC
            true_masks = true_masks[:, :1, :, :]

            y_pred_samples = []
            for i in range(GAUSS_ITERATION):
                with torch.no_grad():
                    logits = net(imgs)
                    y_pred = torch.sigmoid(logits)
                    # y_pred = (y_pred > 0.5).float()
                    y_pred = y_pred[:, :1, :, :]
                    y_pred_samples.append(y_pred[:, 0, :, :])  # y_pred_samples's shape: (inx, bat, H, W )
            y_pred_samples = torch.stack(y_pred_samples, dim=0)
            y_pred_samples = y_pred_samples.type(torch.FloatTensor)
            mean_y_pred = y_pred_samples.mean(dim=0)  # shape: batch, H, W
            # var_y_pred = y_pred_samples.var(dim=0)  # shape: batch, H, w
            ece_values.extend(get_segmentation_mask_uncertainty(mean_y_pred, true_masks))
            pbar.update()
    plt.plot(np.arange(len(ece_values)), ece_values)
    plt.savefig("/data.local/all/hangd/v1/uncertainty1/")

