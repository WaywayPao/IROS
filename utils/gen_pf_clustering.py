import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
import cv2
import torch
import PIL.Image as Image
from collections import OrderedDict
from plantcv import plantcv as pcv
# Set global debug behavior to None (default), "print" (to file), or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = None

USE_GT = True
foldername = "actor_pf_npy" if USE_GT else "pre_cvt_clus_actor_pf_npy"
save_img = True
SAVE_PF = False

data_type = ['interactive', 'non-interactive', 'collision', 'obstacle'][:1]
IMG_H = 100
IMG_W = 200

PIX_PER_METER = 4
sx = IMG_W//2
sy = 3*PIX_PER_METER    # the distance from the ego's center to his head
TARGET = {"roadway":[43,255,123], "roadline":[255,255,255], "vehicle":[120, 2, 255], "pedestrian":[222,134,120]}

# create mask for data preprocessing
# VIEW_MASK_CPU = cv2.imread("./mask_120degree.png")
VIEW_MASK_CPU = np.ones((100, 200, 3), dtype=np.uint8)*255
VIEW_MASK_CPU = (VIEW_MASK_CPU[:100,:,0] != 0).astype(np.float32)
VIEW_MASK = torch.from_numpy(VIEW_MASK_CPU).cuda(0)
VIEW_MASK_IDX = torch.where(VIEW_MASK != 0)
VIEW_MASK_IDX = torch.cat((VIEW_MASK_IDX[0].unsqueeze(-1), VIEW_MASK_IDX[1].unsqueeze(-1)), 1)
VIEW_MASK_Y, VIEW_MASK_X = VIEW_MASK_IDX.T.long()
NEW_VIEW_MASK = torch.zeros((IMG_H, IMG_W, 2)).cuda(0)+0.01
for (y,x) in VIEW_MASK_IDX:
    NEW_VIEW_MASK[y, x] = torch.tensor([y,x])


def get_seg_mask(raw_bev_seg, channel=5):

    """
        src:
            AGENT = 6
            OBSTACLES = 5
            PEDESTRIANS = 4
            VEHICLES = 3
            ROAD_LINE = 2
            ROAD = 1
            UNLABELES = 0
        new (return):
            PEDESTRIANS = 3
            VEHICLES = 2
            ROAD_LINE = 1
            ROAD = 0
    """

    if USE_GT:
        new_bev_seg = np.where((raw_bev_seg<6) & (raw_bev_seg>0), raw_bev_seg, 0)
        new_bev_seg = torch.LongTensor(new_bev_seg)
        one_hot = torch.nn.functional.one_hot(new_bev_seg, channel+1).float()
        
        return one_hot[:,:,1:].numpy()*VIEW_MASK_CPU[:,:,None]
    
    else:
        one_hot = raw_bev_seg*VIEW_MASK_CPU[:,:,None]
        # one_hot = (np.transpose(raw_bev_seg, (1, 2, 0))).astype(np.float32)
        # print(one_hot.shape)

        return one_hot


        # one_hot = np.zeros((IMG_H, IMG_W, channel), dtype=np.float32)

        # for idx, cls in enumerate(TARGET):
        #     target_color = np.array(TARGET[cls])
        #     matching_pixels = np.all(raw_bev_seg == target_color, axis=-1).astype(np.float32)
        #     one_hot[:, :, idx] = matching_pixels*VIEW_MASK_CPU

        # return one_hot
    

def create_roadline_pf(bev_seg, ROBOT_RAD=2.0, KR=400.0):

    road = bev_seg[:, :, 0]
    roadline = bev_seg[:, :, 1]
    vehicle = bev_seg[:, :, 2]
    pedestrian = bev_seg[:, :, 3]
    road = (road+vehicle+pedestrian)!=0

    canny = cv2.Canny(road.astype(np.uint8), 0, 1)
    roadline = torch.from_numpy(roadline + canny)

    oy, ox = torch.where(roadline != 0)
    if len(oy) == 0:
        oy, ox = [0], [0]
    obstacle_tensor = torch.from_numpy(np.stack((oy, ox), 1)).cuda(0)
    
    roadline_pf = create_repulsive_potential(obstacle_tensor, ROBOT_RAD=ROBOT_RAD, KR=KR)

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


def draw_heatmap(data):

    data = np.array(data[::-1])
    data = data.clip(0, 90)
    # print(f"{data.shape} Heat: max {np.max(data)}, min {np.min(data)}, mean {np.mean(data)}")
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.get_cmap('jet'))

    return data


def cal_IOU(clust_masks, points, actor_id=-1, IOU_thres=0.3):
    
    if clust_masks == None:
        return None, -1, None

    max_idx = None
    max_iou = -1

    points = np.array(points)
    zero_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    actor_mask = cv2.fillPoly(zero_mask, [points], color=(255)).astype(np.bool)
    actor_sum = np.sum(actor_mask==1)
    
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    actor_centroid = (sum(x) / len(points), sum(y) / len(points))
    
    # cv2.imwrite(f"mask_{actor_id}.png", actor_mask*255)

    for idx, mask in enumerate(clust_masks):

        mask = mask.copy().astype(np.bool)
        and_cnt = np.sum((actor_mask&mask))
        or_cnt = (np.sum((actor_mask|mask))+1)
        
        iou = and_cnt/or_cnt

        clus_sum = np.sum(mask==1)
        ratio = min(clus_sum, actor_sum)/max(clus_sum, actor_sum)
        y_list, x_list = np.where(mask!=0)
        mask_centroid = (sum(x_list) / len(x_list), sum(y_list) / len(y_list))
        centroid_dis = ((actor_centroid[0]-mask_centroid[0])**2+(actor_centroid[1]-mask_centroid[1])**2)**0.5

        # actor_area = np.sum(actor_mask==1)/or_cnt
        # print(actor_id, idx, actor_area, iou)
        # print(actor_id, idx, actor_centroid, mask_centroid, 
        #       (actor_centroid[0]-mask_centroid[0])**2+(actor_centroid[1]-mask_centroid[1])**2<144, clus_sum, actor_sum, ratio)

        if iou>max_iou and (iou>IOU_thres or (centroid_dis<12 and ratio>0.5)):  # 12pixs == 3meters
            max_iou = iou
            max_idx = idx
    
    # print("result:", actor_id, max_idx, max_iou)
    return max_idx, max_iou, actor_sum


def main(_type, scenario_list, cpu_id=0):
    
    gt_id = 0
    match_id = 0
    total_start = time.time()

    for idx, (basic, variant) in enumerate(sorted(scenario_list), 1):

        variant_path = os.path.join(data_root, basic, "variant_scenario", variant)
        save_npy_folder = os.path.join(save_root, basic, "variant_scenario", variant, foldername)
        bev_box = json.load(open(os.path.join(box_3d_root, basic, "variant_scenario", variant, "bev_box.json")))

        if not os.path.isdir(save_npy_folder):
            os.makedirs(save_npy_folder)

        bev_seg_path = os.path.join(variant_path, "bev-seg")
        start = time.time()

        for seg_frame in sorted(os.listdir(bev_seg_path))[:]:
            frame_id = int(seg_frame.split('.')[0])

            if frame_id != 37:
                continue

            save_npy_path = os.path.join(save_npy_folder,f"{frame_id:08d}.npy")
            # if os.path.isfile(save_npy_path):
            #     continue

            # get bev segmentation
            if USE_GT:
                seg_path = os.path.join(variant_path, "bev-seg", f"{frame_id:08d}.npy")
                raw_bev_seg = np.load(seg_path)
                if save_img:
                    cv2.imwrite("raw_img.png", ((raw_bev_seg/6)*255).astype(np.uint8))
            else:
                seg_path = os.path.join(variant_path, "cvt_bev-seg", seg_frame)
                raw_bev_seg = np.load(seg_path)
                if save_img:
                    for c in range(4):
                        # cv2.imwrite("raw_img.png", 
                        #             ((np.sum(raw_bev_seg, 0).reshape(100,200))/np.max(np.sum(raw_bev_seg, 0))*255).astype(np.uint8))
                        cv2.imwrite(f"raw_img_{c}.png", 
                                    ((raw_bev_seg.reshape(-1, 100,200)[c])*255).astype(np.uint8))
                raw_bev_seg = np.transpose(raw_bev_seg, (1,2,0))

            bev_seg = get_seg_mask(raw_bev_seg[:100])

            # get target point
            goal_frame = f"{frame_id:08d}" if frame_id>=5 else f"{5:08d}" 
            gx, gy = goal_list[basic+'_'+variant][goal_frame]
            gx = sx+gx
            gy = IMG_H-gy
            goal_tensor = torch.tensor([gy, gx]).cuda(0)

            roadline_pf = create_roadline_pf(bev_seg.copy(), ROBOT_RAD=2.0, KR=400.0)
            attractive_pf = create_attractive_pf(goal_tensor)

            save_npy = OrderedDict()
            save_npy['all_actor'] = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            save_npy['roadline'] = roadline_pf.cpu().numpy()
            save_npy['attractive'] = attractive_pf.cpu().numpy()

            obstacle_mask = ((bev_seg[:,:,2]+bev_seg[:,:,3])*255).astype(np.uint8)
            
            if np.sum(obstacle_mask) < 20:
                clust_masks = None
            else:
                _, clust_masks = pcv.spatial_clustering(mask=obstacle_mask, algorithm="DBSCAN", min_cluster_size=5, max_distance=0.5)
                if save_img:
                    # cv2.imwrite("obstacle_mask.png", obstacle_mask)
                    cv2.imwrite(f"clust_img.png", _)
                    print(len(clust_masks), np.sum(obstacle_mask)/255)

            frame_box = bev_box[f"{frame_id:08d}"]
            oy_list = [0]
            ox_list = [0]

            for actor_id in frame_box:
                
                match_clust, iou, _ = cal_IOU(clust_masks, frame_box[actor_id], actor_id, IOU_thres=0.3)
                
                # if _ != None and _ > 100:
                #     gt_id += 1
                #     if match_clust != None:
                #         match_id += 1
                # continue

                if match_clust == None:
                    obstacle_tensor = torch.from_numpy(np.stack(([0], [0]), 1)).cuda(0)
                else:
                    oy, ox = np.where(clust_masks[match_clust]==255)
                    oy_list.extend(oy)
                    ox_list.extend(ox)
                    obstacle_tensor = torch.from_numpy(np.stack((oy, ox), 1)).cuda(0)
                
                actor_pf = create_repulsive_potential(obstacle_tensor, ROBOT_RAD=5.0, KR=1000.0)
                save_npy[actor_id] = actor_pf.cpu().numpy()

                if save_img:
                    plt.close()
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.set_aspect('equal', adjustable='box')
                    plt.xticks([]*IMG_W)
                    plt.yticks([]*IMG_H)
                    img = (save_npy[actor_id]+save_npy['roadline']+save_npy['attractive'])
                    # np.save('teaser.npy', save_npy)
                    cv2.imwrite(f"{basic}-{variant}-{frame_id}-{actor_id}-planning_cv2.png", img/np.max(img)*255)
                    hv = draw_heatmap(img)
                    # plt.plot(sx, sy, "*k")
                    # plt.plot(gx, 100-gy, "*m", markersize=16)
                    plt.axis("equal")
                    plt.savefig(f"{basic}-{variant}-{frame_id}-{actor_id}-planning.png", dpi=300, bbox_inches='tight')
                    # exit()

            obstacle_tensor = torch.from_numpy(np.stack((oy_list, ox_list), 1)).cuda(0)    
            all_actor_pf = create_repulsive_potential(obstacle_tensor, ROBOT_RAD=5.0, KR=1000.0)
            save_npy['all_actor'] = all_actor_pf.cpu().numpy()

            if SAVE_PF:
                np.save(save_npy_path, save_npy)

        end = time.time()
        print(
            f"cpu:{cpu_id:2d}\t{idx:3d}/{len(scenario_list):3d}\t{_type+'_'+basic+'_'+variant}\ttime: {end-start:.2f}s")

    total_end = time.time()
    print(
        f"CPU ID: {cpu_id:3d} finished!!!\tTotal time: {total_end-total_start:.2f}s")
    print("Match id:", match_id, "\tTotal GT:", gt_id, f"({match_id/gt_id*100:.2f}%)")

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')

    train_town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"] # 1350, (45, 30)
    val_town = ["5_"]
    test_town = ["10", "A6", "B3"]   # 515, (47, 11)
    town = test_town

    for _type in data_type:
        
        data_root = os.path.join(
                "/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data", _type)
        
        save_root = os.path.join(
                "/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data", _type)
 
        box_3d_root = os.path.join(
            f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data", _type)
        
        
        if USE_GT:
            goal_list = json.load(open(f"./target_point_{_type}.json"))
        else:
            goal_list = json.load(open(f"../component/TP_model/tp_prediction/{_type}_2024-2-29_232906.json"))

        scenario_list = []

        for basic in sorted(os.listdir(data_root)):
            if not basic[:2] in town:
                continue

            basic_path = os.path.join(data_root, basic, "variant_scenario")

            for variant in sorted(os.listdir(basic_path)):
                # if not (basic == "10_s-8_0_p_j_f_1_0" and variant == "HardRainSunset_high_"):
                #     continue
                # if not (basic == "10_t3-6_1_p_j_f_1_j" and variant == "MidRainyNoon_high_"):
                #     continue
                # if not (basic == "10_i-1_1_c_f_f_1_rl" and variant == "ClearSunset_low_"):
                #     continue
                if not (basic == "10_t2-2_0_c_l_r_1_0" and variant == "CloudySunset_low_"):
                    continue
                # if not (basic == "7_t1-4_0_t_f_r_1_0" and variant == "ClearSunset_low_"):
                #     continue
                # if not (basic == "1_s-4_0_m_l_f_1_s" and variant == "CloudySunset_low_"):
                #     continue
                # if not (basic == "2_t2-2_1_t_u_r_1_0" and variant == "ClearSunset_low_"):
                #     continue

                scenario_list.append((basic, variant))


        main(_type, scenario_list)

        # from multiprocessing import Pool
        # from multiprocessing import cpu_count

        # cpu_n = 15
        # variant_per_cpu = len(scenario_list)//cpu_n+1
        # pool_sz = cpu_count()

        # with Pool(pool_sz) as p:
        #     res = p.starmap(main, [(_type, scenario_list[i*variant_per_cpu:i*variant_per_cpu+variant_per_cpu], i)
        #                     for i in range(cpu_n)])
        #     # for ret in res:
        #     #     print(res)

        #     p.close()
        #     p.join()

