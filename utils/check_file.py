import os
import json
import random
import torch
import numpy as np

data_type = ["interactive", "non-interactive", "obstacle", "collision"][0:1]
undone_list = []

train_town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"] # 1350, (45, 30)
test_town = ["10", "A6", "B3"]   # 515, (47, 11)

is_file = False
# tgt_name = "bev-seg"
# tgt_name = "pre_cvt_actor_pf_npy"
tgt_name = "pre_cvt_clus_actor_pf_npy"
# tgt_name = "actor_pf_npy"
town = test_town

# is_file = True
# tgt_name = "bev_box.json"
# town = train_town+test_town

def main(_type, st=None, ed=None, town=['10', 'B3', 'A6'], cpu_id=0):

    data_root = os.path.join(
        "/media/waywaybao_cs10/DATASET/RiskBench_Dataset", _type)
    tgt_root = os.path.join(
        "/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data", _type)
    # tgt_root = os.path.join(
    #     "/media/waywaybao_cs10/Disk_2/other/new_seg_RiskBench", _type)

    scenario_list = []
    basic_cnt, variant_cnt = 0, 0
    
    for basic in sorted(os.listdir(data_root)):
        if not basic[:2] in town:
            continue
        basic_cnt += 1
        basic_path = os.path.join(data_root, basic, "variant_scenario")

        for variant in sorted(os.listdir(basic_path)):
            # if [_type, basic, variant] in skip_list:
            #     continue
            if os.listdir(basic_path+'/'+variant) == 0:
                print(basic, variant)
            variant_cnt += 1
            scenario_list.append((basic, variant))

    undone = 0
    for idx, (basic, variant) in enumerate(sorted(scenario_list)[st:ed], 1):

        img_path = os.path.join(data_root, basic, "variant_scenario", variant, "rgb/front")
        tgt_path = os.path.join(tgt_root, basic, "variant_scenario", variant, tgt_name)

        miss_tgt = False
        if is_file:
            if not os.path.isfile(tgt_path):
                miss_tgt = True
        else:
            if not os.path.isdir(tgt_path):
                miss_tgt = True
            elif len(os.listdir(tgt_path)) != len(os.listdir(img_path)):
                miss_tgt = True

        if miss_tgt:
            undone += 1
            # print(basic, variant, f"{tgt_N}/{TOTAL_N}")
            undone_list.append([_type, basic, variant])


    print(f"{_type}\tCPU ID:{cpu_id:3d}\t{st:4d}-{ed:4d}\tUndone: {undone:3d}")


if __name__ == '__main__':

    cpu_n = 1
    variant_per_cpu = 3000
    
    for _type in data_type:
        for cpu_id in range(cpu_n):
            main(_type, cpu_id*variant_per_cpu, cpu_id*variant_per_cpu+variant_per_cpu, town, cpu_id)
    
    print(f'Total Undone: {len(undone_list)}')
    # random.shuffle(undone_list)
    # with open('undone_list.json', 'w') as f:
    #     json.dump(undone_list, f, indent=4)

