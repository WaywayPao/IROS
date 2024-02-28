import numpy as np
import json
import os
import csv
from collections import OrderedDict
save_result = True

town = ["10", "A6", "B3"]
dataset_root = "/media/waywaybao_cs10/DATASET/RiskBench_Dataset"
data_types = ['interactive', 'obstacle', 'non-interactive', 'collision'][:]
THRESHOLD_DIS = 15


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
                    idx_tracking == int(frame_id))][:, 1] % 65536
                actor_data = json.load(open(actor_data_path+"/"+frame))

                nearest_id = -1
                nearest_dis = float('inf')

                new_result[basic+"_"+variant][frame_id] = OrderedDict()

                if len(cur_ids) == 0:
                    continue

                for actor_id in cur_ids:

                    if str(actor_id) in actor_data:
                        dis = actor_data[str(actor_id)]["distance"]
                    elif str(actor_id+65536) in actor_data:
                        dis = actor_data[str(actor_id+65536)]["distance"]
                    else:
                        print(data_type, basic, variant, frame_id, actor_id)
                        dis = float('inf')
                        # continue

                    # optional
                    new_result[basic+"_" +
                                variant][frame_id][str(nearest_id)] = False


                    # new_result[basic+"_" +
                    #            variant][frame_id][str(actor_id)] = dis < THRESHOLD_DIS

                    if dis < nearest_dis and dis < THRESHOLD_DIS:
                        nearest_dis = dis
                        nearest_id = int(actor_id)

                if nearest_id != -1:
                    # nearest_id = cur_ids[0]

                    new_result[basic+"_" +
                            variant][frame_id][str(nearest_id)] = True


    return new_result


if __name__ == '__main__':

    for data_type in data_types:

        new_result = main(data_type)

        if save_result:
            with open(f"{data_type}.json", 'w') as f:
                json.dump(new_result, f, indent=4)
