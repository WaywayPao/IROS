import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
import sys
import cv2
import torch
import PIL.Image as Image
from collections import OrderedDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

data_type = ["interactive", "non-interactive", "obstacle", "collision"][0:4]
IMG_H = 100
IMG_W = 200

PIX_PER_METER = 4       # fixed
sx = IMG_W//2
sy = 3*PIX_PER_METER    # the distance from the ego's center to his head


def get_img_coord(R, ego_loc, actor_cord):

    def related_distance(points):
        '''
            Image coordinate system, related to top-left corner (0, 0)
        '''
        points = np.array(points)-ego_loc
        related_dis = points@R

        _gx, _gy = related_dis
        gx = max(0, min(IMG_W-1, _gx*PIX_PER_METER+sx))
        gy = max(0, min(IMG_H-1, IMG_H-(_gy*PIX_PER_METER)))

        return [int(gx+0.5), int(gy+0.5)]

    cord_list = np.array(
        [related_distance(actor_cord[f"cord_{i}"][:2]) for i in range(0, 8, 2)])

    p1, p2, p3, p4 = cord_list.tolist()
    cord_list = [p1, p2, p4, p3]
    return cord_list


def get_actor_traj(actor_id, frame_data):

    actor_traj_list = []

    if actor_id in frame_data and "cord_bounding_box" in frame_data[actor_id]:
        actor_cord = frame_data[actor_id]["cord_bounding_box"]
        actor_traj_list.append(actor_cord)

    return actor_traj_list


def main(_type, scenario_list, cpu_id=0):


    total_start = time.time()

    for idx, (basic, variant) in enumerate(sorted(scenario_list), 1):

        start = time.time()

        basic_path = os.path.join(data_root, basic, "variant_scenario")
        variant_path = os.path.join(basic_path, variant)
        save_dir = os.path.join(save_root, basic, "variant_scenario", variant)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "bev_box.json")

        if os.path.isfile(save_path):
            continue

        actors_data = OrderedDict()
        egos_data = OrderedDict()
        bev_box = OrderedDict()

        actor_data_path = os.path.join(variant_path, "actors_data")
        ego_data_path = os.path.join(variant_path, "ego_data")
        tracklet = np.load(variant_path+"/tracking.npy")
        if len(tracklet) == 0:
            idx_tracking = None
        else:
            idx_tracking = tracklet[:, 0]

        for frame in sorted(os.listdir(ego_data_path)):
            frame_path = os.path.join(actor_data_path, frame)
            actors_data[frame] = json.load(open(frame_path))

            frame_path = os.path.join(ego_data_path, frame)
            egos_data[frame] = json.load(open(frame_path))

        
        for frame in sorted(os.listdir(ego_data_path))[:]:
            frame_id = int(frame.split('.')[0])

            if idx_tracking is None:
                cur_ids = []
            else:
                cur_ids = tracklet[np.where(idx_tracking == int(frame_id))][:, 1]

                ##############
                ego_data = egos_data[frame]
                theta = ego_data["compass"]
                theta = np.array(theta*np.pi/180.0)
                # clockwise
                R = np.array([[np.cos(theta), np.sin(theta)],
                                [np.sin(theta), -np.cos(theta)]])
                ego_loc = np.array(
                    [ego_data["location"]["x"], ego_data["location"]["y"]])
                ##############

            box_dict = OrderedDict()
            for actor_id in cur_ids:

                actor_traj_list = get_actor_traj(
                    str(actor_id), frame_data=actors_data[f"{frame_id:08d}.json"])
                if len(actor_traj_list) == 0:
                    continue

                cord_list = get_img_coord(R, ego_loc, actor_traj_list[0])
                box_dict[str(actor_id)] = cord_list
            
            bev_box[f"{frame_id:08d}"] = box_dict


        with open(save_path, "w") as f:
            json.dump(bev_box, f, indent=4)


        end = time.time()
        print(f"cpu:{cpu_id:2d}\t{idx:3d}/{len(scenario_list):3d}\t{_type+'_'+basic+'_'+variant}\ttime: {end-start:.2f}s")

    total_end = time.time()
    print(
        f"CPU ID: {cpu_id:3d} finished!!!\tTotal time: {total_end-total_start:.2f}s")


if __name__ == '__main__':


    train_town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"]
    test_town = ["10", "A6", "B3"]
    town = train_town+test_town


    for _type in data_type:

        data_root = os.path.join(
            "/media/waywaybao_cs10/DATASET/RiskBench_Dataset", _type)
        save_root = os.path.join(
            f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data", _type)
        scenario_list = []

        for basic in sorted(os.listdir(data_root)):
            if not basic[:2] in town:
                continue

            basic_path = os.path.join(data_root, basic, "variant_scenario")
            for variant in sorted(os.listdir(basic_path)):
                scenario_list.append((basic, variant))

        main(_type, scenario_list)
