from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from fetch_data_for_next_phase import get_pool_data


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, train=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = get_pool_data("data_one32nd_std.json")
        # print("\n\nInitialized .....\nID: \n", self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        pil_img = pil_img.resize((256, 256))
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255.0

        return img_trans

    def __getitem__(self, i):

        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        # print("\n\nindex i: ", i)
        # print("self.ids: ", self.ids[i])
        # print("mask file: ", mask_file)
        # print("img_file: ", img_file)

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
