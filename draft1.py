from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split


# ailab
dir_img = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data/imgs/'
dir_mask = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data/masks/'

dir_img_test = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data_test/imgs/'
dir_mask_test = '/data.local/all/hangd/src_code_3/Pytorch-UNet/data_test/masks/'

dir_img_draft = '/DATA/hangd/cardi/RobustSegmentation/data_draft/imgs/'
dir_mask_draft = '/DATA/hangd/cardi/RobustSegmentation/data_draft/masks/'


dataset = BasicDataset(dir_img, dir_mask, 1)
data_test = BasicDataset(imgs_dir=dir_img_test, masks_dir=dir_mask_test, train=False, scale=1)
train = dataset
batch_size = 4
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
train_loader_un_shuffle = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
                         drop_last=True)

img = next(iter(train_loader))['image']
mask = next(iter(train_loader))['mask']
print("train_loader: ", img.shape, mask.shape)

