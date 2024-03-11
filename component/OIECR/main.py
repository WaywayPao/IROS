import numpy as np
import json
import os
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict

save_img = False
SAVE_RESULT = True
STEP = 1
K = 1

# src_wp_name = f"./results/new_testing_last_waypoints_list_step={STEP}.json"
# save_name = f"./results/last_wp_raw_score_step={STEP}.json"

src_wp_name = f"./results/new_gt_waypoints_list_step={STEP}.json"
save_name = f"./results/wp_gt_raw_score_step={STEP}.json"

data_types = ['interactive', 'non-interactive', 'collision', 'obstacle'][:1]


def save_roi_json(score_dict, max_score=None, json_name=None):

    new_rp_dict = OrderedDict()

    for sample, score in score_dict.items():
        scenario, frame_id, actor_id = sample.split('#')

        if not scenario in new_rp_dict:
            new_rp_dict[scenario] = OrderedDict()
        if not f"{int(frame_id)}" in new_rp_dict[scenario]:
            new_rp_dict[scenario][f"{int(frame_id)}"] = OrderedDict()

        new_rp_dict[scenario][f"{int(frame_id)}"][actor_id] = score/max_score

    with open(f"./{json_name}", "w") as f:
        json.dump(new_rp_dict, f, indent=4)


def cal_importance_score(ego_wp, removal_wp, k=K):

    def cal_RS(wp1, wp2):
        RS = 0

        for i in range(k):
            RS += ((wp1[i][0]-wp2[i][0])**2+(wp1[i][1]-wp2[i][1])**2)**0.5
        return RS


    def cal_VS():
        VS = 0
        # TODO
        return VS

    RS = cal_RS(ego_wp, removal_wp)
    VS = cal_VS()

    IS = max(RS, VS)

    return IS


def main():

    """
        "10_s-8_0_p_j_f_1_0_HardRainSunset_high_": {
            "23": {
                "all_actor": [...],
                "17981#remove": [...],
                "19466#remove": [...],
            },
            "24":{
                "all_actor": [...],
                ...
            },
        }
    """    
    score_dict = OrderedDict()
    src_wp_list = json.load(open(src_wp_name))
    max_score = 0
    
    for scenario in src_wp_list:
        for frame_id in src_wp_list[scenario]:

            ego_wp = src_wp_list[scenario][frame_id]["all_actor"]

            for actor_id in src_wp_list[scenario][frame_id]:
                if actor_id == "all_actor":
                    continue
                else:
                    removal_wp = src_wp_list[scenario][frame_id][actor_id]
                    actor_id = actor_id.split('#')[0]
                    raw_score = cal_importance_score(ego_wp, removal_wp)
                    sample = scenario+'#'+frame_id+'#'+actor_id
                    score_dict[sample] = raw_score
                    
                    if raw_score > max_score:
                        max_score = raw_score

        # print(scenario, "Done")

    if SAVE_RESULT:
        save_roi_json(score_dict, max_score=max_score, json_name=save_name)


if __name__ == '__main__':
    main()
