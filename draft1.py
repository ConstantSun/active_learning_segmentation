from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from dynamic_dataloader import RestrictedDataset, BasicDataset
from fetch_data_for_next_phase import get_pool_data

# ailab
dir_img = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data/imgs/'
dir_mask = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data/masks/'

dir_img_test = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data_test/imgs/'
dir_mask_test = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data_test/masks/'

dir_img_draft = '/DATA/hangd/cardi/RobustSegmentation/data_draft/imgs/'
dir_mask_draft = '/DATA/hangd/cardi/RobustSegmentation/data_draft/masks/'

pool_data = get_pool_data("data_one32nd_category.json")
# dataset = RestrictedDataset(dir_img, dir_mask, pool_data, True)
dataset = BasicDataset(dir_img, dir_mask, pool_data)
pool_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

batch = next(iter(pool_loader))
img = batch['image']
mask = batch['mask']
id = batch['id']
print("train_loader: ", img.shape, mask.shape)
print("id: ", id)
