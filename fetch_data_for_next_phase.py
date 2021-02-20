import torch
from tqdm import tqdm
import torchvision
from dynamic_dataloader import RestrictedDataset
from torch.utils.data import DataLoader, random_split

import json
from std_metric import get_segmentation_mask_uncertainty
import acquisition_function
import random

# ailab
dir_img = "/data.local/all/hangd/dynamic_data/full/data/imgs/"
dir_mask = '/data.local/all/hangd/dynamic_data/full/data/masks/'

dir_img_test = '/data.local/all/hangd/dynamic_data/full/data_test/imgs/'
dir_mask_test = '/data.local/all/hangd/dynamic_data/full/data_test/masks/'

global GAUSS_ITERATION
GAUSS_ITERATION = 30


def add_image_id_to_pool(id: str, filename="pooling_data.json"):
    """id: image name, e.g: GEMS_IMG__2010_MAR__12__HA122541__F8HB4A50_24"""
    with open(filename, 'r+') as f:
        dic = json.load(f)
        dic["ids"].append(id)
    with open(filename, 'w') as file:
        json.dump(dic, file)


def delete_image_id_from_pool(id: str, filename="pooling_data.json"):
    """id: image name, e.g: GEMS_IMG__2010_MAR__12__HA122541__F8HB4A50_24"""
    with open(filename, 'r+') as f:
        dic = json.load(f)
        dic["ids"].remove(id)
    with open(filename, 'w') as file:
        json.dump(dic, file)


def get_pool_data(filename="pooling_data.json"):
    """return a list of image names (image id)"""
    with open(filename, 'r+') as f:
        dic = json.load(f)
        return dic["ids"]


def update_training_pool_ids(net: torch.nn, training_pool_ids_path: str, all_training_data, device:str):
    """
    training_pool_ids_path: the path to json file which contains images id in training pool.
    This function will use an acquisition function to collect new 100 imgs into training pool each phase.
        /Increase the json file 100 more imgs each phase.
    """
    batch_size = 1
    training_pool_data = get_pool_data(training_pool_ids_path)
    all_training_data = get_pool_data(all_training_data)
    active_pool = set(all_training_data) - set(training_pool_data)
    dataset = RestrictedDataset(dir_img, dir_mask, list(active_pool))
    pool_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    std = []
    imgs_id = []

    net.eval()
    n_pool = len(dataset)
    with tqdm(total=n_pool, desc='STD calculating', unit='batch', leave=False) as pbar:
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
            _std = get_segmentation_mask_uncertainty(std_y_pred)
            _imgs_id = batch['id']
            for i in range(batch_size):
                if i >= len(_std):
                    continue
                std.extend(_std)
                imgs_id.extend(_imgs_id)
            pbar.update()

    std, imgs_id = zip(*sorted(zip(std, imgs_id))) # order = ascending
    print("length of std/imgs_id: ", len(std), len(imgs_id))
    top_100img = imgs_id[-100:]
    for i in top_100img:
        add_image_id_to_pool(i, training_pool_ids_path)
    print("Adding successfully!")


def update_training_pool_ids_2(net: torch.nn, training_pool_ids_path: str, all_training_data,
                               device: str, acquisition_func: str = "cfe"):
    """
    training_pool_ids_path: the path to json file which contains images id in training pool.
    acquisition_func: string name of acquisition function:
                    available function: mutual_information, mean_first_entropy, category_first_entropy
    This function will use an acquisition function to collect new 100 imgs into training pool each phase.
        /Increase the json file 100 more imgs each phase.

    """
    batch_size = 1
    training_pool_data = get_pool_data(training_pool_ids_path)
    all_training_data = get_pool_data(all_training_data)
    active_pool = set(all_training_data) - set(training_pool_data)
    dataset = RestrictedDataset(dir_img, dir_mask, list(active_pool))
    pool_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    value = []
    imgs_id = []

    if acquisition_func == "cfe":
        evaluation_criteria = acquisition_function.category_first_entropy
    elif acquisition_func == "mfe":
        evaluation_criteria = acquisition_function.mean_first_entropy
    elif acquisition_func == "mi":
        evaluation_criteria = acquisition_function.mutual_information
    else:
        print("Error choosing acquisition function")
        evaluation_criteria = None

    net.eval()
    n_pool = len(dataset)
    with tqdm(total=n_pool, desc='STD calculating', unit='batch', leave=False) as pbar:
        for ind, batch in enumerate(tqdm(pool_loader)):
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32) # BHWC
            true_masks = true_masks[:, :1, :, :]

            _value = evaluation_criteria(GAUSS_ITERATION, net, imgs)
            _imgs_id = batch['id']
            for i in range(batch_size):
                if i >= len(_value):
                    continue
                value.extend(_value)
                imgs_id.extend(_imgs_id)
            pbar.update()

    value, imgs_id = zip(*sorted(zip(value, imgs_id))) # order = ascending
    print("length of value/imgs_id: ", len(value), len(imgs_id))
    top_100img = imgs_id[-100:] # the higher
    for i in top_100img:
        add_image_id_to_pool(i, training_pool_ids_path)
    print("Adding successfully!")


def update_training_pool_ids_random(training_pool_ids_path: str, all_training_data: str):
    """
    training_pool_ids_path:  path to json file which is the training pool.
    all_training_data: path to json file which is
    """
    with open(all_training_data, 'r+') as f:
        dic = json.load(f)
        ids = dic["ids"]
        random_eles = random.sample(ids, 100)
        for i in random_eles:
            add_image_id_to_pool(i, training_pool_ids_path)

