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

data_type = ["interactive", "non-interactive", "obstacle", "collision"][1:4]
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

    if actor_id in frame_data:
        actor_cord = frame_data[actor_id]["cord_bounding_box"]
        actor_traj_list.append(actor_cord)

    return actor_traj_list



def main(_type, st=None, ed=None, town=['10', 'B3', 'A6'], cpu_id=0):

    data_root = os.path.join(
        "/media/waywaybao_cs10/DATASET/RiskBench_Dataset", _type)
    save_root = os.path.join(
        f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data", _type)
    skip_list = json.load(open("./skip_scenario.json"))

    scenario_list = []

    for basic in sorted(os.listdir(data_root)):
        if not basic[:2] in town:
            continue

        basic_path = os.path.join(data_root, basic, "variant_scenario")

        for variant in sorted(os.listdir(basic_path)):
            if [_type, basic, variant] in skip_list:
                continue
            # if not (basic == "10_t2-2_0_c_l_r_1_0" and variant == "CloudySunset_low_"):
            #     continue
            # if not (basic == "1_t2-2_1_t_u_r_1_0" and variant == "ClearSunset_mid_"):
            #     continue
            # if not (basic == "1_s-4_0_m_l_f_1_s" and variant == "CloudySunset_low_"):
            #     continue
            scenario_list.append((basic, variant))
    
    total_start = time.time()

    for idx, (basic, variant) in enumerate(sorted(scenario_list)[st:ed], 1):

        basic_path = os.path.join(data_root, basic, "variant_scenario")
        variant_path = os.path.join(basic_path, variant)

        actor_data_path = os.path.join(variant_path, "actors_data")
        ego_data_path = os.path.join(variant_path, "ego_data")
        tracklet = np.load(variant_path+"/tracking.npy")
        idx_tracking = tracklet[:, 0]

        actors_data = OrderedDict()
        egos_data = OrderedDict()

        for frame in sorted(os.listdir(ego_data_path)):
            frame_path = os.path.join(actor_data_path, frame)
            actors_data[frame] = json.load(open(frame_path))

            frame_path = os.path.join(ego_data_path, frame)
            egos_data[frame] = json.load(open(frame_path))

        start = time.time()
        bev_box = OrderedDict()
        
        for frame in sorted(os.listdir(ego_data_path))[:]:
            frame_id = int(frame.split('.')[0])
            
            # print(basic, variant, frame)
            # if frame_id != 39:
            #     continue

            # import cv2
            # img = np.load(os.path.join(save_root, basic, "variant_scenario", variant, "bev-seg", f"{frame_id:08d}.npy"))
            # cv2.imwrite("./tmp_ego.png", (img==6).astype(np.uint8)*255)
            # exit()

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


        save_dir = os.path.join(save_root, basic, "variant_scenario", variant)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "bev_box.json")

        # save_path = os.path.join("./bev_box.json")
        with open(save_path, "w") as f:
            json.dump(bev_box, f, indent=4)


        end = time.time()
        print(
            f"cpu:{cpu_id:2d}\t{st+idx:3d}/{ed:3d}\t{_type+'_'+basic+'_'+variant}\ttime: {end-start:.2f}s")

    total_end = time.time()
    print(
        f"CPU ID: {cpu_id:3d} finished!!!\tTotal time: {total_end-total_start:.2f}s")


if __name__ == '__main__':

    from multiprocessing import Pool
    from multiprocessing import cpu_count

    train_town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"] # 1350, (45, 30)
    test_town = ["10", "A6", "B3"]   # 515, (47, 11)
    town = train_town+test_town
    cpu_n = 1
    variant_per_cpu = 2000    ###

    for _type in data_type:

        # pool_sz = cpu_count()

        # with Pool(pool_sz) as p:
        #     res = p.starmap(main, [(_type, i*variant_per_cpu, i*variant_per_cpu+variant_per_cpu, town, i)
        #                     for i in range(cpu_n)])
        #     # for ret in res:
        #     #     print(res)

        #     p.close()
        #     p.join()

        main(_type, 0, 2000, town)
