from fetch_data_for_next_phase import add_image_id_to_pool
import json
import random


def fetch_random(filename="pooling_data.json", k=100):
    with open(filename, 'r+') as f:
        dic = json.load(f)
        ids = dic["ids"]
        random_eles = random.sample(ids, k)
        for i in random_eles:
            add_image_id_to_pool(i, "data_one32nd_random.json")
            # print(i)


fetch_random()
