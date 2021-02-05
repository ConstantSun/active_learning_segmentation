import glob
from shutil import copyfile
from tqdm import tqdm

img_path = "/data.local/all/hangd/src_code_3/Pytorch-UNet/data/imgs/*"
dest_img_path = "/data.local/all/hangd/dynamic_data/one32rd/imgs"
dest_mask_path = "/data.local/all/hangd/dynamic_data/one32rd/masks"

count = 0
num_first_trial_files = len(glob.glob(img_path))//32

for img in tqdm(glob.glob(img_path)):
    # print(img)
    count += 1
    copyfile(img, dest_img_path+img[img.rfind("/"):])
    mask_path = img.replace("/imgs/", "/masks/")
    copyfile(mask_path, dest_mask_path+mask_path[mask_path.rfind("/"):])
    if count == num_first_trial_files:
        break
