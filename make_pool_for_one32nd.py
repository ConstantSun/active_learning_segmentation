import os
import glob
from tqdm import tqdm
from fetch_data_for_next_phase import add_image_id_to_pool

path_fulldata = "/data.local/all/hangd/src_code_3/Pytorch-UNet/data/"
path_one32nd = "/data.local/all/hangd/dynamic_data/one32rd/"

for file in tqdm(glob.glob(path_fulldata+"imgs/*")):
    id = file[file.rfind("/"):]
    if str(path_one32nd + "imgs" + id) not in glob.glob(path_one32nd+"imgs/*"):
        add_image_id_to_pool(id[1:][:-4])


