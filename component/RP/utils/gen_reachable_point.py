import numpy as np
import json
import os
from collections import OrderedDict

SAVE_RESULT = True
# foldername = "pre_cvt_clus_actor_pf_npy"
# save_name = "testing_wo_roadline_reachable_point"
# save_name = "testing_keep_reachable_point"
# save_name = "testing_reachable_point"
# goal_list = json.load(open(f"../../TP_model/tp_prediction/interactive_2024-2-29_232906.json"))

foldername = "actor_pf_npy"
save_name = "training_reachable_point"
goal_list = json.load(open(f"../../../utils/target_point_interactive.json"))

sample_root = f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset"
data_root = "/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data"

train_town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"] # 1350, (45, 30)
val_town = ["5_"]
test_town = ["10", "A6", "B3"]   # 515, (47, 11)
town = train_town+val_town+test_town
# town = test_town

data_types = ['interactive', 'non-interactive', 'collision', 'obstacle'][:1]
PIX_PER_METER = 4
IMG_H = 100
IMG_W = 200
Sx = IMG_W//2
Sy = IMG_H-3*PIX_PER_METER    # the distance from the ego's center to his head


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
                        scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+"all_actor")

                    actor_id = str(sample[1])
                    scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+actor_id)
                    cnt += 1

    print(cnt)

    return scenario_list[:]


def find_RP(gx, gy, potential_map, res=1.0):

    init_potential = float('inf')
    start_step = 0
    
    def DFS(ix, iy, min_potential, step, min_idx=None, min_d=float('inf')):

        is_check[iy, ix] = True

        if step > PIX_PER_METER*25:
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
                    min_idx = [inx, iny]

                if d < res*PIX_PER_METER:
                    return min_idx, min_d
                
                elif not is_check[iny, inx]:
                    min_idx, min_d = DFS(inx, iny, min_potential=p, step=step+1, min_idx=min_idx, min_d=min_d)

        return min_idx, min_d

    is_check = np.zeros(potential_map.shape, dtype=bool)
    MOTION = [[1, 0], [0, -1], [1, -1]]
    min_idx1, min_d1 = DFS(ix=Sx, iy=Sy, min_potential=init_potential, step=start_step, min_idx=None, min_d=float('inf'))

    is_check = np.zeros(potential_map.shape, dtype=bool)
    MOTION = [[-1, 0], [0, -1], [-1, -1]]
    min_idx2, min_d2 = DFS(ix=Sx, iy=Sy, min_potential=init_potential, step=start_step, min_idx=None, min_d=float('inf'))

    if min_d1 < min_d2:
        return min_idx1, min_d1
    else:
        return min_idx2, min_d2


def save_roi_json(occupy_dict, json_name):

    new_rp_dict = OrderedDict()

    for scenario, rp in occupy_dict.items():
        data_type, basic, variant, frame_id, actor_id = scenario.split('#')

        if not data_type in new_rp_dict:
            new_rp_dict[data_type] = OrderedDict()
        if not basic+'_'+variant in new_rp_dict[data_type]:
            new_rp_dict[data_type][basic+'_'+variant] = OrderedDict()
        if not f"{int(frame_id)}" in new_rp_dict[data_type][basic+'_'+variant]:
            new_rp_dict[data_type][basic+'_'+variant][f"{int(frame_id)}"] = OrderedDict()
        
        new_rp_dict[data_type][basic+'_'+variant][f"{int(frame_id)}"][actor_id] = [rp[0]-100, 100-rp[1]]

    for data_type in new_rp_dict:
        with open(f"./{json_name}.json", "w") as f:
            json.dump(new_rp_dict[data_type], f, indent=4)


def main():
    
    scenario_list = read_scenario()
    occupy_dict = OrderedDict()
    
    for sample in scenario_list:
        data_type, basic, variant, frame, actor_id = sample.split('#')
        # if actor_id != "all_actor":
        #     continue
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

        if actor_id != "all_actor":
            gt_pf = ((npy_file["all_actor"]-actor_pf)+roadline_pf+attractive_pf).clip(0.1, 90)
            # gt_pf = ((actor_pf)+roadline_pf+attractive_pf).clip(0.1, 90)
        else:
            gt_pf = (actor_pf+roadline_pf+attractive_pf).clip(0.1, 90)
        
        # import cv2
        # print(np.max(gt_pf), gt_pf.shape)
        # cv2.imwrite(f"{actor_id}.png", (gt_pf/90.0*255).astype(np.uint8))
        # gt_pf = (attractive_pf).clip(0.1, 90)

        gx, gy = goal_list[basic+'_'+variant][f"{frame_id:08d}"]
        gx = Sx+gx
        gy = IMG_H-gy

        min_idx, mid_d = find_RP(gx, gy, gt_pf, res=1.0)
        occupy_dict[sample] = min_idx

        print(sample)


    if SAVE_RESULT:
        save_roi_json(occupy_dict, json_name=save_name)


if __name__ == '__main__':
    main()


