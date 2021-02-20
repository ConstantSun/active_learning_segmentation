from pan_regnety120 import PAN
import torch
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torchvision

from eval import eval_net
from fetch_data_for_next_phase import get_pool_data, update_training_pool_ids

from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from dynamic_dataloader import RestrictedDataset
from pathlib import Path

global val_iou_score
global best_val_iou_score
global best_test_iou_score

val_iou_score = 0.
best_val_iou_score = 0.
best_test_iou_score = 0.

# ailab
dir_img = "/data.local/all/hangd/dynamic_data/full/data/imgs/"
dir_mask = "/data.local/all/hangd/dynamic_data/full/data/masks/"

dir_img_test = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data_test/imgs/'
dir_mask_test = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data_test/masks/'


def train_net(
        dir_checkpoint,
        n_classes,
        n_channels,
        device,
        epochs=30,
        save_cp=True,
        img_scale=1):
    global best_val_iou_score
    global best_test_iou_score

    net = PAN()
    net.to(device=device)
    batch_size = 4
    lr = 1e-5
    writer = SummaryWriter(comment=f'_{net.__class__.__name__}_LR_{lr}_BS_{batch_size}_STD_ACQUISITION')
    global_step = 0

    logging.basicConfig(filename="./logging_one32nd_std.txt",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if n_classes > 1 else 'max', patience=2)
    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    num_phases = 25  # total 2689 imgs, within each phase: fetching 100 imgs to training set.
    training_pool_ids_path = "data_one32nd_std.json"
    all_training_data = "data_all.json"

    for phase in range(num_phases):
        # Within a phase, save the best epoch (having highest test_iou) checkpoint and save its test_iou to TF_Board
        #                 also, load the best right previous checkpoint
        selected_images = get_pool_data(training_pool_ids_path)
        data_train = RestrictedDataset(dir_img, dir_mask, selected_images)
        data_test = BasicDataset(imgs_dir=dir_img_test, masks_dir=dir_mask_test, train=False, scale=img_scale)

        train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
                                 drop_last=True)
        right_previous_ckpt_dir = Path(dir_checkpoint + 'ckpt.pth')
        if right_previous_ckpt_dir.is_file():
            net.load_state_dict(
                torch.load(dir_checkpoint + 'ckpt.pth', map_location=device)
            )
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            n_train = len(data_train)
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    imgs = batch['image']
                    true_masks = batch['mask']
                    assert imgs.shape[1] == n_channels, \
                        f'Network has been defined with {n_channels} input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if n_classes == 1 else torch.long

                    true_masks = true_masks.to(device=device, dtype=mask_type)
                    masks_pred = net(imgs)  # return BCHW = 8_1_256_256
                    _tem = net(imgs)
                    # print("IS DIFFERENT OR NOT: ", torch.sum(masks_pred - _tem))
                    true_masks = true_masks[:, :1, :, :]
                    loss = criterion(masks_pred, true_masks)
                    epoch_loss += loss.item()
                    # writer.add_scalar('Loss/train', loss.item(), global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()
                    pbar.update(imgs.shape[0])
                    global_step += 1
            # Tính dice và iou score trên tập Test set, ghi vào tensorboard .
            test_score_dice, test_score_iou = eval_net(net, test_loader, n_classes, device)
            if test_score_iou > best_test_iou_score:
                best_test_iou_score = test_score_iou
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'best_CP_epoch{epoch + 1}_one32th_.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
            logging.info('Test Dice Coeff: {}'.format(test_score_dice))
            print('Test Dice Coeff: {}'.format(test_score_dice))
            writer.add_scalar(f'Phase_{phase}_Dice/test', test_score_dice, epoch)

            logging.info('Test IOU : {}'.format(test_score_iou))
            print('Test IOU : {}'.format(test_score_iou))
            writer.add_scalar(f'Phase_{phase}_IOU/test', test_score_iou, epoch)
        print(f"Phase_{phase}_best iou: ", best_test_iou_score)
        torch.save(net.state_dict(),
                   dir_checkpoint + 'ckpt.pth')
        writer.add_scalar('Phase_IOU/test', best_test_iou_score, phase)
        # Fetching data for next phase - Update pooling images.
        update_training_pool_ids(net, training_pool_ids_path, all_training_data, device)

    writer.close()


def get_args():

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--method', dest='method', type=str, default='i',
                        help='Choose dropout method: i for MCdropout ; w for Dropconnect')
    parser.add_argument('-cuda', '--cuda-inx', type=int, nargs='?', default=1,
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
    dir_ckp = "./check_point_active/"
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

    # For a specific architecture
    try:
        train_net(dir_checkpoint=dir_ckp,
                  n_classes=n_classes,
                  n_channels=n_channels,
                  epochs=args.epochs,
                  device=device,
                  img_scale=args.scale,
                  )
    except KeyboardInterrupt:
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
