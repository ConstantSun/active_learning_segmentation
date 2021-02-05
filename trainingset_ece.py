from pan_regnety120 import PAN
import torch
import argparse
import logging
import os
import sys
import cv2
from baal.bayesian import MCDropoutConnectModule
from matplotlib import pyplot as plt
from ece_metric import *

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torchvision

from eval import eval_net
from visualize import visualize_to_tensorboard

from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp

global val_iou_score
global best_val_iou_score
global best_test_iou_score

val_iou_score = 0.
best_val_iou_score = 0.
best_test_iou_score = 0.

# ailab
dir_img = "/data.local/all/hangd/dynamic_data/imgs/"
dir_mask = '/data.local/all/hangd/dynamic_data/masks/'

dir_img_test = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data_test/imgs/'
dir_mask_test = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data_test/masks/'

global GAUSS_ITERATION
GAUSS_ITERATION = 30

def train_net(
        dir_checkpoint,
        n_classes,
        bilinear,
        n_channels,
        device,
        epochs=30,
        val_percent=0.1,
        save_cp=True,
        img_scale=1):

    global best_val_iou_score
    global best_test_iou_score

    net = PAN()
    ckpt_path = "/data.local/all/hangd/v1/uncertainty1/best_CP_epoch15_test_iou_85_with_25_percent_original_training_dataset.pth"
    net.to(device=device)
    net.load_state_dict(
        torch.load(ckpt_path, map_location=device)
    )
    logging.info(f'Model loaded from {ckpt_path}')

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    data_test = BasicDataset(imgs_dir=dir_img_test, masks_dir=dir_mask_test, train=False, scale=img_scale)

    batch_size = 4
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    lr = 1e-5
    writer = SummaryWriter(comment=f'_{net.__class__.__name__}_ECE_on_25percentage_data')
    global_step = 0

    logging.info(f'''Starting training:

        Device:          {device.type}
    ''')

    epochs = 1
    ece_values = []
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        n_train = len(dataset)
        with tqdm(total=n_train, desc='Validation round', unit='batch', leave=False) as pbar:
            for batch in train_loader:
                imgs, true_masks = batch['image'], batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32) # BHWC
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
                ece_values.extend(get_segmentation_mask_uncertainty(mean_y_pred, true_masks))
                pbar.update()

    for inx, ece_val in enumerate(ece_values):
        writer.add_scalar("ECE_on_quarter_of_training_set", ece_val, inx)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--method', dest='method', type=str, default='i',
                        help='Choose dropout method: i for MCdropout ; w for Dropconnect')
    parser.add_argument('-cuda', '--cuda-inx', type=int, nargs='?', default=0,
                        help='index of cuda', dest='cuda_inx')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=30,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    dir_ckp = "/data.local/all/hangd/v1/uncertainty1/"

    if torch.cuda.is_available():
        _device = 'cuda:' + str(args.cuda_inx)
    else:
        _device = 'cpu'
    device = torch.device(_device)
    logging.info(f'Using device {device}')

    n_classes = 1
    n_channels = 3
    bilinear = True

    logging.info(f'Network:\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if bilinear else "Transposed conv"} upscaling')

    try:
        train_net(dir_checkpoint=dir_ckp,
                  n_classes=n_classes,
                  bilinear=bilinear,
                  n_channels=n_channels,
                  epochs=args.epochs,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
