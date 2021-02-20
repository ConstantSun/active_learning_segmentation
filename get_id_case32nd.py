import glob
from tqdm import tqdm
from fetch_data_for_next_phase import add_image_id_to_pool

# path_one32nd = "/data.local/all/hangd/dynamic_data/one32rd/"
path_all = "/data.local/all/hangd/dynamic_data/full/data/"


for file in tqdm(glob.glob(path_all+"imgs/*")):
    id = file[file.rfind("/"):]
    add_image_id_to_pool(id[1:][:-4], "data_all.json")


