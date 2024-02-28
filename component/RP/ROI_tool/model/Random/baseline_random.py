import numpy as np
import json
import os
import time
import random
from collections import OrderedDict

save_result = True
town = ["10", "A6", "B3"]
dataset_root = "/media/waywaybao_cs10/DATASET/RiskBench_Dataset"
data_types = ['interactive', 'non-interactive', 'collision', 'obstacle'][:]
ODDS = 50
random.seed(time.time())

def main(data_type):

    _type_path = os.path.join(dataset_root, data_type)
    new_result = OrderedDict()
    skip_list = json.load(open("../../../datasets/skip_scenario.json"))

    for basic in sorted(os.listdir(_type_path)):

        if not basic[:2] in town:
            continue
        basic_path = os.path.join(_type_path, basic, "variant_scenario")

        for variant in sorted(os.listdir(basic_path)):

            if [data_type, basic, variant] in skip_list:
                continue

            actor_data_path = os.path.join(
                basic_path, variant, "actors_data")

            tracking = np.load(os.path.join(
                basic_path, variant, "tracking.npy"))
            idx_tracking = tracking[:, 0]

            new_result[basic+"_"+variant] = OrderedDict()
            for frame in sorted(os.listdir(actor_data_path)):

                frame_id = str(int(frame.split('.')[0]))

                cur_ids = tracking[np.where(
                    idx_tracking == int(frame_id))][:, 1]%65536

                new_result[basic+"_"+variant][frame_id] = OrderedDict()

                if len(cur_ids) > 0:
                    for actor_id in cur_ids:
                        new_result[basic+"_" +
                                variant][frame_id][str(actor_id)] = random.randint(1, 100) <= ODDS

                    # risky_id = random.choice(cur_ids)
                    # new_result[basic+"_" +
                    #         variant][frame_id][str(risky_id)] = True

    return new_result


if __name__ == '__main__':

    for data_type in data_types:

        new_result = main(data_type)

        if save_result:
            with open(f"{data_type}.json", 'w') as f:
                json.dump(new_result, f, indent=4)
