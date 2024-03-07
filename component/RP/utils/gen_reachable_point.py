import numpy as np
import json
import os
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict

save_img = False
SAVE_RESULT = True

train_town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"] # 1350, (45, 30)
val_town = ["5_"]
test_town = ["10", "A6", "B3"]   # 515, (47, 11)

foldername = "pre_cvt_clus_actor_pf_npy"
# save_name = "testing_wo_roadline_reachable_point"
# save_name = "testing_keep_reachable_point"
save_name = "new_testing_reachable_point"
goal_list = json.load(open(f"../../TP_model/tp_prediction/interactive_2024-2-29_232906.json"))
town = test_town

# foldername = "actor_pf_npy"
# save_name = "new_testing_gt_reachable_point"
# goal_list = json.load(open(f"../../../utils/target_point_interactive.json"))
# # town = train_town+val_town+test_town
# town = test_town

VIEW_MASK_CPU = cv2.imread("../../../utils/mask_120degree.png")
VIEW_MASK_CPU = (VIEW_MASK_CPU[:100,:,0] != 0).astype(np.float32)

sample_root = f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset"
data_root = "/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data"

data_types = ['interactive', 'non-interactive', 'collision', 'obstacle'][:1]
PIX_PER_METER = 4
IMG_H = 100
IMG_W = 200
Sx = IMG_W//2
Sy = IMG_H-4*PIX_PER_METER    # the distance from the ego's center to his head


def read_scenario():

    scenario_list = []
    cnt = 0

    for _type in data_types:
    
        type_path = os.path.join(sample_root, _type)

        for basic in sorted(os.listdir(type_path))[:]:
            if not basic[:2] in town:
                continue
            basic_path = os.path.join(type_path, basic, "variant_scenario")

            for variant in sorted(os.listdir(basic_path)):


                variant_path = os.path.join(basic_path, variant)
                tracking_path = os.path.join(variant_path, "tracking.npy")
                tracking_npy = np.load(tracking_path)

                frame_id = 0
                for sample in tracking_npy:
                    if sample[0] < 5:
                        continue

                    if frame_id != sample[0]:
                        frame_id = int(sample[0])
                        scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+"all_actor"+"#")
                        scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+"no_actor"+"#")

                    # if not (basic == "10_s-8_0_p_j_f_1_0" and variant == "HardRainSunset_high_" and frame_id == 23):
                    #     continue
                    # if not (basic == "10_i-1_1_c_f_f_1_rl" and variant == "ClearSunset_low_" and frame_id == 37):
                    #     continue
                    # if not (basic == "1_s-4_0_m_l_f_1_s" and variant == "CloudySunset_low_" and frame_id == 55):
                    #     continue
                    # if not (basic == "2_t2-2_1_t_u_r_1_0" and variant == "ClearSunset_low_" and frame_id == 5):
                    #     continue

                    actor_id = str(sample[1])
                    scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+actor_id+"#keep")
                    scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+actor_id+"#remove")
                    scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+actor_id+"#no_road_keep")
                    scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+actor_id+"#no_road_remove")
                    cnt += 1

    print("GT ID:", cnt)
    print("Testing Sample:", len(scenario_list))

    return scenario_list[:]


def _find_RP(gx, gy, potential_map, res=1.0):

    init_potential = float('inf')
    start_step = 0
    
    def DFS(ix, iy, min_potential, step, min_idx=[], min_d=float('inf')):

        is_check[iy, ix] = True

        if step > PIX_PER_METER*20:
            return min_idx, min_d
        
        for i in range(len(MOTION)):
            motion = MOTION[i]
            inx = int(ix + motion[0])
            iny = int(iy + motion[1])

            if iny >= len(potential_map) or inx >= len(potential_map[0]) or inx < 0 or iny < 0:
                p = float('inf')  # outside area
            else:
                p = potential_map[iny, inx]

            if p < min_potential:
                d = np.hypot(gx - inx, gy - iny)
                if d < min_d:
                    min_d = d
                    min_idx.append([inx, iny])

                if d < res*PIX_PER_METER:
                    return min_idx, min_d
                
                elif not is_check[iny, inx]:
                    min_idx, min_d = DFS(inx, iny, min_potential=p, step=step+1, min_idx=min_idx, min_d=min_d)

        return min_idx, min_d

    MOTION = [[1, 0], [0, 1], [-1, 0], [1, 1],
              [-1, 1], [0, -1], [-1, -1], [1, -1]]
    
    is_check = np.zeros(potential_map.shape, dtype=bool)
    # MOTION = [[1, 0], [0, -1], [1, -1]]
    min_idx1, min_d1 = DFS(ix=Sx, iy=Sy, min_potential=init_potential, step=start_step, min_idx=[], min_d=float('inf'))
    return min_idx1, min_d1

    is_check = np.zeros(potential_map.shape, dtype=bool)
    MOTION = [[-1, 0], [0, -1], [-1, -1]]
    min_idx2, min_d2 = DFS(ix=Sx, iy=Sy, min_potential=init_potential, step=start_step, min_idx=None, min_d=float('inf'))

    if min_d1 < min_d2:
        return min_idx1, min_d1
    else:
        return min_idx2, min_d2


def find_RP(gx, gy, potential_map, res=1.0):

    ix = Sx
    iy = Sy
    point_list = [[ix, iy]]

    MOTION = [[1, 0], [0, 1], [-1, 0], [1, 1],
              [-1, 1], [0, -1], [-1, -1], [1, -1]]

    iter = 0

    while iter < PIX_PER_METER*20:
        iter += 1
        min_potential = float('inf')
        minix, miniy = -1, -1

        for i in range(len(MOTION)):
            inx = int(ix + MOTION[i][0])
            iny = int(iy + MOTION[i][1])
            
            if iny >= len(potential_map) or inx >= len(potential_map[0]) or inx < 0 or iny < 0 or VIEW_MASK_CPU[iny, inx] == 0:
                p = float('inf')  # outside area
            else:
                p = potential_map[iny, inx]
            if p < min_potential:
                min_potential = p
                minix = inx
                miniy = iny

        ix = minix
        iy = miniy
        xp = ix
        yp = iy
        d = np.hypot(gx - xp, gy - yp)
        point_list.append([xp, yp])

        if d < res*PIX_PER_METER:
            return point_list, point_list[-1]

    return point_list, point_list[-1]


def save_roi_json(occupy_dict, json_name):

    new_rp_dict = OrderedDict()

    for scenario, rp in occupy_dict.items():
        data_type, basic, variant, frame_id, actor_id, mode = scenario.split('#')

        if not data_type in new_rp_dict:
            new_rp_dict[data_type] = OrderedDict()
        if not basic+'_'+variant in new_rp_dict[data_type]:
            new_rp_dict[data_type][basic+'_'+variant] = OrderedDict()
        if not f"{int(frame_id)}" in new_rp_dict[data_type][basic+'_'+variant]:
            new_rp_dict[data_type][basic+'_'+variant][f"{int(frame_id)}"] = OrderedDict()

        if mode in ["keep", "remove", "no_road_keep", "no_road_remove"]:
            new_rp_dict[data_type][basic+'_'+variant][f"{int(frame_id)}"][actor_id+"#"+mode] = [rp[0]-100, 100-rp[1]]
        else:   # actor_id in ["all_actor", "no_actor"]
            new_rp_dict[data_type][basic+'_'+variant][f"{int(frame_id)}"][actor_id] = [rp[0]-100, 100-rp[1]]

    for data_type in new_rp_dict:
        with open(f"./{json_name}.json", "w") as f:
            json.dump(new_rp_dict[data_type], f, indent=4)


def draw_heatmap(data):

    data = np.array(data[::-1])
    data = data.clip(0, 90)
    # print(f"{data.shape} Heat: max {np.max(data)}, min {np.min(data)}, mean {np.mean(data)}")
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.get_cmap('jet'))

    return data


def main():
    
    scenario_list = read_scenario()
    occupy_dict = OrderedDict()
    
    for sample in scenario_list:
        data_type, basic, variant, frame, actor_id, mode = sample.split('#')

        ####################################
        # if actor_id != "all_actor":
        #     continue
        ####################################

        variant_path = os.path.join(data_root, data_type, basic, "variant_scenario", variant)
        frame_id = int(frame)

        pf_path = os.path.join(variant_path, foldername, f"{frame_id:08d}.npy")
        npy_file = np.load(pf_path, allow_pickle=True).item()

        if actor_id in npy_file:
            actor_pf = npy_file[actor_id]
        else:
            actor_pf = np.zeros((100,200))

        roadline_pf = npy_file['roadline']
        attractive_pf = npy_file['attractive']

        if actor_id == "all_actor":
            gt_pf = (actor_pf+roadline_pf+attractive_pf).clip(0.1, 90)
        elif actor_id == "no_actor":
            gt_pf = (roadline_pf+attractive_pf).clip(0.1, 90)
        elif mode == "keep":
            gt_pf = (actor_pf+roadline_pf+attractive_pf).clip(0.1, 90)
        elif mode == "remove":
            gt_pf = ((npy_file["all_actor"]-actor_pf)+roadline_pf+attractive_pf).clip(0.1, 90)
        elif mode == "no_road_keep":
            gt_pf = (actor_pf+attractive_pf).clip(0.1, 90)
        elif mode == "no_road_remove":
            gt_pf = ((npy_file["all_actor"]-actor_pf)+attractive_pf).clip(0.1, 90)

        # import cv2
        # print(np.max(gt_pf), gt_pf.shape)
        # cv2.imwrite(f"{actor_id}.png", (gt_pf/90.0*255).astype(np.uint8))
        # gt_pf = (attractive_pf).clip(0.1, 90)

        gx, gy = goal_list[basic+'_'+variant][f"{frame_id:08d}"]
        gx = Sx+gx
        gy = IMG_H-gy

        min_idx, mid_d = find_RP(gx, gy, gt_pf, res=1.0)

        if save_img:
            plt.close()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal', adjustable='box')
            plt.xticks([]*IMG_W)
            plt.yticks([]*IMG_H)
            img = gt_pf
            cv2.imwrite(f"{basic}-{variant}-{frame_id}-{actor_id}-planning_cv2.png", img/np.max(img)*255)
            hv = draw_heatmap(img)
            # print(actor_id, gx, 100-gy, min_idx[0], 100-min_idx[1])
            # plt.plot(min_idx[0], 100-min_idx[1], "*k", markersize=16)
            for i in range(len(min_idx)):
                print(actor_id, gx, 100-gy, min_idx[i][0], 100-min_idx[i][1])
                plt.plot(min_idx[i][0], 100-min_idx[i][1], "*k", markersize=4)
            plt.plot(gx, 100-gy, "*m", markersize=16)
            plt.axis("equal")
            plt.savefig(f"{basic}-{variant}-{frame_id}-{actor_id}-planning.png", dpi=300, bbox_inches='tight')
            # exit()
        
        occupy_dict[sample] = mid_d

        print(sample)


    if SAVE_RESULT:
        save_roi_json(occupy_dict, json_name=save_name)


if __name__ == '__main__':
    main()


