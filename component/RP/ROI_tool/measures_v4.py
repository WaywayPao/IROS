import numpy as np
import os
import json
import argparse
from collections import OrderedDict

data_type = ['interactive', 'non-interactive', 'collision', 'obstacle'][:1]
SAVE_ROI = True
threshold = 20.0


def cal_dis(tgt_xy, src_xy):
    return (tgt_xy[0]-src_xy[0])**2+(tgt_xy[1]-src_xy[1])**2


def main(_type, new_dict_path, topk=1, EPS=1e-8):

    new_dict = json.load(open(new_dict_path))
    roi_result = OrderedDict()

    for basic in new_dict:
        for variant in new_dict[basic]:

            roi_result[basic+"_"+variant] = OrderedDict()
            for frame in new_dict[basic][variant]:

                roi_result[basic+"_"+variant][frame] = OrderedDict()
                frame_info = new_dict[basic][variant][frame]
                actor_0_RP = frame_info["0"]


                ### for using topk
                # FDE_dict = OrderedDict()
                # for actor_id in frame_info:
                #     if actor_id == "0":
                #         continue
                #     roi_result[basic+"_"+variant][frame][actor_id] = False
                #     actor_RP = frame_info[actor_id]
                #     FDE = cal_dis(actor_0_RP, actor_RP)**0.5
                #     FDE_dict[actor_id] = FDE

                # sorted_keys = list(
                #     dict(sorted(FDE_dict.items(), key=lambda x: x[1])[::-1]).keys())
                
                # for actor_id in FDE_dict:
                #     score = min(1., FDE_dict[actor_id]/5.0)
                #     roi_result[basic+"_"+variant][frame][actor_id] = (score >= threshold)


                ######################################
                # for actor_id in sorted_keys[:topk]:
                #     # if len(sorted_keys) > 1:
                #     #     is_risky = FDE_dict[actor_id]-FDE_dict[sorted_keys[-1]] > threshold
                #     # else:
                #     #     is_risky = True
                #     is_risky = FDE_dict[actor_id] > threshold
                #     roi_result[basic+"_"+variant][frame][actor_id] = is_risky

                ######################################
                for actor_id in frame_info:
                    if actor_id == "0":
                        continue

                    actor_RP = frame_info[actor_id]
                    is_risky = cal_dis(actor_0_RP, actor_RP) > threshold
                    roi_result[basic+"_"+variant][frame][actor_id] = is_risky
                    # roi_result[basic+"_"+variant][frame][actor_id] = cal_dis(actor_0_RP, actor_RP)**0.5


    return roi_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, required=True)
    args = parser.parse_args()

    threshold = args.threshold
    print(f"threshold: {threshold}")
    new_dict_path = "../results/new_dict.json"
    save_path = "./model_transpose/pf_40x20"

    for _type in data_type:
        roi_result = main(_type, new_dict_path)
        with open(f"{save_path}/{_type}.json", 'w') as f:
            json.dump(roi_result, f, indent=4)

