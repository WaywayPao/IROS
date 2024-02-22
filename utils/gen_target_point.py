import numpy as np
import os
import time
import json
from collections import OrderedDict

data_type = ['interactive', 'non-interactive', 'obstacle', 'collision'][:3]
IMG_H = 100
IMG_W = 200

SAVE_JSON = False
EGO_TARGET_FRAME = 60   # fixed
PIX_PER_METER = 4       # fixed
sx = IMG_W//2
sy = 3*PIX_PER_METER    # the distance from the ego's center to his head


def get_target_point(egos_data, cur_frame_id, N, R, ego_loc):

    def related_coordinate(related_vec):
        '''
            Cartesian coordinate system, related to ego (0, 0)
        '''
        related_dis_rotation = related_vec@R

        _gx, _gy = related_dis_rotation
        gx = _gx*PIX_PER_METER
        gy = _gy*PIX_PER_METER+sy

        return int(gx+0.5), int(gy+0.5)


    for target_frame_id in range(min(cur_frame_id+EGO_TARGET_FRAME, N), N+1):
        
        target_ego_data = egos_data[f"{target_frame_id:08d}.json"]

        target_loc = [target_ego_data["location"]["x"], target_ego_data["location"]["y"]]
        related_vec = np.array(target_loc)-ego_loc

        if (related_vec[0]**2+related_vec[1]**2) < 100 and target_frame_id < N:
            continue
        
        gx, gy = related_coordinate(related_vec)
        break

    return gx, gy


def main(_type, town=['10', 'B3', 'A6']):

    tp_list = OrderedDict()

    total_start = time.time()
    for idx, (basic, variant) in enumerate(sorted(scenario_list), 1):

        basic_path = os.path.join(data_root, basic, "variant_scenario")
        variant_path = os.path.join(basic_path, variant)

        ego_data_path = os.path.join(variant_path, "ego_data")
        egos_data = OrderedDict()
        tp_list[basic+'_'+variant] = OrderedDict()
        N = len(os.listdir(ego_data_path))

        for frame in sorted(os.listdir(ego_data_path)):
            frame_path = os.path.join(ego_data_path, frame)
            egos_data[frame] = json.load(open(frame_path))
        
        for frame in sorted(os.listdir(ego_data_path)):

            frame_id = int(frame.split('.')[0])
            ego_data = egos_data[frame]
            theta = ego_data["compass"]
            theta = np.array(theta*np.pi/180.0)

            # clockwise
            R = np.array([[np.cos(theta), np.sin(theta)],
                            [np.sin(theta), -np.cos(theta)]])
            ego_loc = np.array(
                [ego_data["location"]["x"], ego_data["location"]["y"]])

            gx, gy = get_target_point(
                egos_data, cur_frame_id=frame_id, N=N, R=R, ego_loc=ego_loc)

            tp_list[basic+'_'+variant][f"{frame_id:08d}"] = [gx, gy]

        print(f"{idx:4d}/{len(scenario_list):4d}\t{_type+'_'+basic+'_'+variant}\t fininshed!")

    total_end = time.time()
    print(f"Total time: {total_end-total_start:.2f}s")
    return tp_list


if __name__ == '__main__':

    train_town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"]
    test_town = ["10", "A6", "B3"]
    town = train_town+test_town

    for _type in data_type:

        data_root = os.path.join(
            "/media/waywaybao_cs10/DATASET/RiskBench_Dataset", _type)
        scenario_list = []

        for basic in sorted(os.listdir(data_root)):
            if not basic[:2] in town:
                continue
            basic_path = os.path.join(data_root, basic, "variant_scenario")

            for variant in sorted(os.listdir(basic_path)):
                scenario_list.append((basic, variant))

        tp_list = main(_type, town)

        if SAVE_JSON:
            with open(f"./target_point_{_type}.json", "w") as f:
                json.dump(tp_list, f, indent=4)