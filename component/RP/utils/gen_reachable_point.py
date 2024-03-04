import numpy as np
import json
import os
from collections import OrderedDict


sample_root = f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset"
data_root = "/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data"
goal_list = json.load(open(f"../component/TP_model/tp_prediction/interactive_2024-2-29_232906.json"))

train_town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"] # 1350, (45, 30)
val_town = ["5_"]
test_town = ["10", "A6", "B3"]   # 515, (47, 11)
town = test_town

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
    
        data_root = os.path.join(sample_root, _type)

        for basic in sorted(os.listdir(data_root)):
            if not basic[:2] in town:
                continue
            basic_path = os.path.join(data_root, basic, "variant_scenario")

            for variant in sorted(os.listdir(basic_path)):
                variant_path = os.path.join(data_root, basic, "variant_scenario", variant)
                tracking_path = os.path.join(variant_path, "tracking.npy")
                tracking_npy = np.load(tracking_path)

                frame_id = 0
                for sample in tracking_npy:
                    if sample[0] < 5:
                        continue

                    if frame_id != sample[0]:
                        frame_id = str(sample[0])
                        scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+"all_actor")

                    actor_id = str(sample[1])
                    scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+actor_id)
                    cnt += 1

    print(cnt)

    return scenario_list


def occcupy(gx, gy, potential_map, res=1.0):

    init_potential = float('inf')
    start_step = 0
    occupy_npy = np.zeros(potential_map.shape, dtype=bool)
    is_check = np.zeros(potential_map.shape, dtype=bool)

    def DFS(ix, iy, min_potential, step):
        
        is_check[ix, iy] = True
        if step > PIX_PER_METER*10:
            return
        
        for i in range(len(MOTION)):
            motion = MOTION[i]
            inx = int(ix + motion[0])
            iny = int(iy + motion[1])

            if inx >= len(potential_map) or iny >= len(potential_map[0]):
                p = float('inf')  # outside area
            else:
                p = potential_map[inx, iny]

            if p < min_potential:
                d = np.hypot(gx - inx, gy - iny)
                occupy_npy[inx, iny] = True

                if d < res*PIX_PER_METER:
                    return 
                elif not is_check[inx, iny]:
                    DFS(inx, iny, min_potential=p, step=step+1)

    MOTION = [[1, 0], [0, 1], [-1, 0], [1, 1],
            [-1, 1], [0, -1], [-1, -1], [1, -1]][:-3]
    DFS(ix=Sx, iy=Sy, min_potential=init_potential, step=start_step)

    return occupy_npy


def save_roi_json(roi_occupy_dictdict, json_name):

    new_tp_dict = OrderedDict()

    for scenario, pred_stop in roi_dict.items():
        data_type, basic, variant, frame_id, actor_id = scenario.split('#')

        if not data_type in new_tp_dict:
            new_tp_dict[data_type] = OrderedDict()
        if not basic+'_'+variant in new_tp_dict[data_type]:
            new_tp_dict[data_type][basic+'_'+variant] = OrderedDict()
        if not f"{int(frame_id)}" in new_tp_dict[data_type][basic+'_'+variant]:
            new_tp_dict[data_type][basic+'_'+variant][f"{int(frame_id)}"] = OrderedDict()
        
        new_tp_dict[data_type][basic+'_'+variant][f"{int(frame_id)}"][actor_id] = pred_stop

    for data_type in new_tp_dict:
        with open(f"./{json_name}.json", "w") as f:
            json.dump(new_tp_dict[data_type], f, indent=4)


def main():
    
    scenario_list = read_scenario()
    occupy_dict = OrderedDict()
    
    for sample in scenario_list:
        data_type, basic, variant, frame, actor_id = sample.split('#')

        variant_path = os.path.join(sample_root, data_type, basic, "variant_scenario", variant)
        frame_id = int(frame)

        pf_path = os.path.join(variant_path, "pre_cvt_clus_actor_pf_npy", f"{frame_id:08d}.npy")
        npy_file = np.load(pf_path, allow_pickle=True).item()

        if actor_id in npy_file:
            actor_pf = npy_file[actor_id]
        else:
            actor_pf = np.zeros((100,200))

        roadline_pf = npy_file['roadline']
        attractive_pf = npy_file['attractive']

        if actor_id != "all_actor":
            gt_pf = ((npy_file["all_actor"]-actor_pf)+roadline_pf+attractive_pf).clip(0.1, 90)
        else:
            gt_pf = (actor_pf+roadline_pf+attractive_pf).clip(0.1, 90)

        gx, gy = goal_list[basic+'_'+variant][f"{frame_id:08d}"]
        gx = Sx+gx
        gy = IMG_H-gy

        occupy_npy = occcupy(gx, gy, gt_pf, res=1.0)
        occupy_dict[sample] = occupy_npy
    

    save_roi_json(occupy_dict, json_name="...json")

if __name__ == '__main__':
    main()


