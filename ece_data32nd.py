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
from dynamic_dataloader1 import RestrictedDataset
from torch.utils.data import DataLoader, random_split
from std_metric import get_segmentation_mask_uncertainty
import json
from dataset import BasicDataset


# ailab
dir_img = "/data.local/all/hangd/dynamic_data/one32rd/imgs/"
dir_mask = '/data.local/all/hangd/dynamic_data/one32rd/masks/'

dir_img_test = '/data.local/all/hangd/dynamic_data/full/data_test/imgs/'
dir_mask_test = '/data.local/all/hangd/dynamic_data/full/data_test/masks/'

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
    ckpt_path = "/data.local/all/hangd/v1/uncertainty1/best_CP_epoch29_one32th_.pth"
    net.to(device=device)
    net.load_state_dict(
        torch.load(ckpt_path, map_location=device)
    )
    writer = SummaryWriter(comment=f'_{net.__class__.__name__}_ece_one32nd_training_set')

    logging.info(f'Model loaded from {ckpt_path}')
    batch_size = 4

    dataset = BasicDataset(dir_img, dir_mask, True)
    pool_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # data_test = BasicDataset(imgs_dir=dir_img_test, masks_dir=dir_mask_test, train=False, scale=img_scale)
    # test_loader = DataLoader(data_test, batch_size=16, shuffle=False, num_workers=2, pin_memory=True,drop_last=True)

    logging.info(f'''Starting selecting in pool:
        Device:          {device.type}
    ''')

    epochs = 1
    # test_score_dice, test_score_iou = eval_net(net, test_loader, n_classes, device)
    # print(f"TEST iou = {test_score_iou}, dice = {test_score_dice} ")
    std = []
    for epoch in range(epochs):
        net.eval()
        epoch_loss = 0
        n_pool = len(dataset)
        with tqdm(total=n_pool, desc='ECE calculating', unit='batch', leave=False) as pbar:
            for ind, batch in enumerate(tqdm(pool_loader)):
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
                std_y_pred = y_pred_samples.std(dim=0)    # shape: batch, H, W
                grid = torchvision.utils.make_grid(mean_y_pred.unsqueeze(1))
                writer.add_image('images', grid, ind)
                _std = get_segmentation_mask_uncertainty(std_y_pred)
                std.extend(_std)

    for ind, val in enumerate(std):
        writer.add_scalar("std in pool - case one32nd data", val, ind)

    writer.add_histogram("Histogram std corresponding", np.array(std), 1)
    std = torch.cuda.FloatTensor(std)
    mean = std.mean()
    _std = std.std()
    writer.add_scalar("Mean std", mean, 1)
    writer.add_scalar("STD std", _std, 1)
    print("Mean: ", mean)
    print("std: ", _std)

                # is_selected = False
                # # TODO
                # # Using some constraint here to select data
                # # Acquisition functions
                # if is_selected:
                #     id = batch['id']
                #     add_image_id_to_pool(id)
                # pbar.update()


def get_args():
    parser = argparse.ArgumentParser(description='Fetching dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
