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

data_type = ['interactive', 'non-interactive', 'collision', 'obstacle'][:1]
IMG_H = 100
IMG_W = 200

save_img = True
SAVE_PF = False

PIX_PER_METER = 4       # fixed
# PIX_PER_METER = 200/46.63   # fixed
sx = IMG_W//2
sy = 3*PIX_PER_METER    # the distance from the ego's center to his head
TARGET = {"roadway":[43,255,123], "roadline":[255,255,255], "pedestrian":[222,134,120], "vehicle":[255,2,120]}

# create mask for data preprocessing
# VIEW_MASK = cv2.imread("/media/waywaybao_cs10/Disk_2/new_seg_RiskBench/VIEW_MASK.png")[:100,:,0]
VIEW_MASK = np.ones((IMG_H, IMG_W)) # GT
VIEW_MASK = torch.from_numpy(VIEW_MASK != 0).cuda(0)
VIEW_MASK_IDX = torch.where(VIEW_MASK != 0)
VIEW_MASK_IDX = torch.cat((VIEW_MASK_IDX[0].unsqueeze(-1), VIEW_MASK_IDX[1].unsqueeze(-1)), 1)
VIEW_MASK_Y, VIEW_MASK_X = VIEW_MASK_IDX.T.long()
NEW_VIEW_MASK = torch.zeros((IMG_H, IMG_W, 2)).cuda(0)+0.01
for (y,x) in VIEW_MASK_IDX:
    NEW_VIEW_MASK[y, x] = torch.tensor([y,x])


def get_seg_mask(raw_bev_seg, channel=4):

    assert 1==0, "TBD"
    """
        AGENT = 6
        OBSTACLES = 5
        PEDESTRIANS = 4
        VEHICLES = 3
        # crosswalk
        ROAD_LINE = 2
        ROAD = 1
        UNLABELES = 0
    """

    # bev_seg = np.zeros((IMG_H, IMG_W, channel), dtype=np.float32)
    # raw_bev_seg = raw_bev_seg[:100]

    # for idx, cls in enumerate(TARGET):
    #     target_color = np.array(TARGET[cls])
    #     matching_pixels = np.all(raw_bev_seg == target_color, axis=-1)[:100].astype(np.float32)
    #     bev_seg[:, :, idx] = matching_pixels

    bev_seg = np.zeros((IMG_H, IMG_W, channel), dtype=np.float32)
    raw_bev_seg = raw_bev_seg[:100]
    bev_seg[:, :, 0] = raw_bev_seg==0
    bev_seg[:, :, 1] = raw_bev_seg==1

    return bev_seg


def create_roadline_pf(road, roadline):

    roadline_pf = torch.zeros((IMG_H, IMG_W), dtype=torch.float32).cuda(0)

    canny = cv2.Canny(road.astype(np.uint8), 0, 1)
    roadline = torch.from_numpy(roadline + canny)

    oy, ox = torch.where(roadline != 0)
    obstacle_tensor = torch.cat((oy.unsqueeze(-1), ox.unsqueeze(-1)), 1).cuda(0)
    
    roadline_pf = create_repulsive_potential(obstacle_tensor, ROBOT_RAD=2.0, KR=400.0)

    return roadline_pf


def create_attractive_pf(goal_tensor, KP=0.75, EPS=0.01):

    # attractive_pf = KP * np.hypot(x - gx, y - gy)
    attractive_pf = (KP*torch.sum((NEW_VIEW_MASK - goal_tensor)**2, axis=2)**0.5) * VIEW_MASK

    return attractive_pf


def create_repulsive_potential(obstacle_tensor, ROBOT_RAD=5.0, KR=1000.0):
    
    repulsive_pf = torch.zeros((IMG_H, IMG_W), dtype=torch.float32).cuda(0)

    # find the minimum distance min_dq to obstacle for each pixel
    dis_list = torch.sum((VIEW_MASK_IDX[:, None, :] - obstacle_tensor)**2, axis=2)
    min_dq = (torch.min(dis_list, axis=1)[0])**0.5

    # if min_dq is smaller than the given radius, a repulsive force is assigned, otherwise 0
    # repulsive force = KR / ((min_dq + 2) ** 2)
    mask = min_dq <= ROBOT_RAD*10
    repulsive_pf[VIEW_MASK_Y[mask], VIEW_MASK_X[mask]] = KR / ((min_dq[mask] + 2) ** 2)

    return repulsive_pf


def get_obstacle(R, ego_loc, actor_traj_list):

    ox = [0]
    oy = [0]

    def related_distance(points):
        '''
            Cartesian coordinate system, related to ego (0, sx)
        '''
        points = np.array(points)-ego_loc
        related_dis = points@R

        _gx, _gy = related_dis
        gx = _gx*PIX_PER_METER+sx
        gy = _gy*PIX_PER_METER

        return [gx, gy]


    is_used = np.zeros((IMG_H, IMG_W), dtype=bool)
    for actor_cord in actor_traj_list:

        cord_list = np.array(
            [related_distance(actor_cord[f"cord_{i}"][:2]) for i in range(0, 8, 2)])
        min_x = min(max(int(np.min(cord_list[:, 0])), 1), IMG_W)
        max_x = min(max(int(np.max(cord_list[:, 0])), 1), IMG_W)
        min_y = min(max(int(np.min(cord_list[:, 1])), 1), IMG_H)
        max_y = min(max(int(np.max(cord_list[:, 1])), 1), IMG_H)

        polygon = Polygon([cord_list[0],
                           cord_list[1],
                           cord_list[3],
                           cord_list[2]])

        for gx in range(min_x, max_x):
            for gy in range(min_y, max_y):
                if not is_used[gy, gx] and polygon.contains(Point(gx, gy)):
                    is_used[gy, gx] = True
                    ox.append(gx)
                    oy.append(IMG_H-gy)

    return ox, oy


def get_actor_traj(actor_id, frame_data):

    actor_traj_list = []

    if actor_id in frame_data:
        actor_cord = frame_data[actor_id]["cord_bounding_box"]
        actor_traj_list.append(actor_cord)

    return actor_traj_list


def draw_heatmap(data):

    data = np.array(data[::-1])
    data = data.clip(0, 90)
    # print(f"{data.shape} Heat: max {np.max(data)}, min {np.min(data)}, mean {np.mean(data)}")
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.get_cmap('jet'))

    return data


def main(_type, st=None, ed=None, town=['10', 'B3', 'A6'], cpu_id=0):

    data_root = os.path.join(
        "/media/waywaybao_cs10/DATASET/RiskBench_Dataset", _type)
    save_root = os.path.join(
        f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data", _type)
    skip_list = json.load(open("./skip_scenario.json"))
    goal_list = json.load(open("./goal_list.json"))

    scenario_list = []

    for basic in sorted(os.listdir(data_root)):
        if not basic[:2] in town:
            continue

        basic_path = os.path.join(data_root, basic, "variant_scenario")

        for variant in sorted(os.listdir(basic_path)):
            if [_type, basic, variant] in skip_list:
                continue
            if not (basic == "10_t2-2_0_c_l_r_1_0" and variant == "CloudySunset_low_"):
                continue
            # if not (basic == "1_s-4_0_m_l_f_1_s" and variant == "CloudySunset_low_"):
            #     continue
            scenario_list.append((basic, variant))
    

    total_start = time.time()

    for idx, (basic, variant) in enumerate(sorted(scenario_list)[st:ed], 1):

        basic_path = os.path.join(data_root, basic, "variant_scenario")
        variant_path = os.path.join(basic_path, variant)
        save_npy_folder = os.path.join(save_root, basic, "variant_scenario", variant, "actor_pf_npy")

        if not os.path.isdir(save_npy_folder):
            os.makedirs(save_npy_folder)

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

        for frame in sorted(os.listdir(ego_data_path))[:]:

            frame_id = int(frame.split('.')[0])
            
            # print(basic, variant, frame)
            if frame_id != 39:
                continue
            save_npy_path = os.path.join(save_npy_folder,f"{frame_id:08d}.npy")
            # if os.path.isfile(save_npy_path):
            #     continue
			
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


            save_npy = OrderedDict()

            # camera_name = f"{frame_id:08d}.png"
            # camera_path = os.path.join(save_root, basic, "variant_scenario", variant, "bev-seg", camera_name)
            # raw_bev_seg = np.array(Image.open(camera_path).convert('RGB').copy())
            # bev_seg = get_seg_mask(raw_bev_seg)
            npy_name = f"{frame_id:08d}.npy"
            npy_path = os.path.join(save_root, basic, "variant_scenario", variant, "bev-seg", npy_name)
            raw_bev_seg = np.load(npy_path)
            bev_seg = get_seg_mask(raw_bev_seg)
            
            roadline_pf = create_roadline_pf(road=bev_seg[:, :,0], roadline=bev_seg[:, :,1])

            gx, gy = goal_list[basic+'_'+variant][f"{frame_id:08d}"]
            # gx = (gx-80)*4/PIX_PER_METER+100
            # gy = 100-(gy-0)*4/PIX_PER_METER
            gx = sx+gx
            gy = IMG_H-gy

            goal_tensor = torch.tensor([gy, gx]).cuda(0)
            attractive_pf = create_attractive_pf(goal_tensor)

            save_npy['all_actor'] = torch.zeros((IMG_H, IMG_W), dtype=torch.float32).cuda(0)
            save_npy['roadline'] = roadline_pf
            save_npy['attractive'] = attractive_pf

            for actor_id in cur_ids:

                actor_traj_list = get_actor_traj(
                    str(actor_id), frame_data=actors_data[f"{frame_id:08d}.json"])

                if len(actor_traj_list) == 0:
                    continue

                ox, oy = get_obstacle(R, ego_loc, actor_traj_list)
                obstacle_tensor = torch.from_numpy(np.stack((oy, ox), 1)).cuda(0)
                actor_pf = create_repulsive_potential(obstacle_tensor, ROBOT_RAD=5.0, KR=1000.0)
                save_npy[str(actor_id)] = actor_pf
                save_npy['all_actor'] = torch.where(save_npy['all_actor']>actor_pf, save_npy['all_actor'], actor_pf)

                print(actor_id)
            
            if save_img:

                plt.close()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect('equal', adjustable='box')
                plt.xticks([]*IMG_W)
                plt.yticks([]*IMG_H)
                img = (save_npy['all_actor']).cpu().numpy()
                cv2.imwrite(f"{basic}-{variant}-{frame_id}-{actor_id}-planning_cv2.png", img/np.max(img)*255)
                hv = draw_heatmap(img)
                plt.plot(sx, sy, "*k")
                plt.plot(gx, 100-gy, "*m")
                plt.axis("equal")
                plt.savefig(f"{basic}-{variant}-{frame_id}-{actor_id}-planning.png", dpi=300, bbox_inches='tight')
                # exit()

            if SAVE_PF:
                np.save(save_npy_path, save_npy)

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
    variant_per_cpu = 1865    ###

    for _type in data_type:

        # pool_sz = cpu_count()

        # with Pool(pool_sz) as p:
        #     res = p.starmap(main, [(_type, i*variant_per_cpu, i*variant_per_cpu+variant_per_cpu, town, i)
        #                     for i in range(cpu_n)])
        #     # for ret in res:
        #     #     print(res)

        #     p.close()
        #     p.join()

        main(_type, 0, 1865, town)
