import numpy as np
import os
import json

from collections import OrderedDict

save_result = True
data_root = "./"
data_type = ['interactive', 'non-interactive', 'collision', 'obstacle'][:]


def read_data(data_type, method, attr=""):

    json_path = os.path.join(data_root, method, f"{data_type}.json")

    roi_file = open(json_path)
    roi_result = json.load(roi_file)
    roi_file.close()

    return roi_result


def ROI_transpose(method, roi_result, threshold=0.5, topk=None):

    new_result = OrderedDict()
    
    for scenario_weather in roi_result.keys():
        new_result[scenario_weather] = OrderedDict()

        for frame_id in roi_result[scenario_weather]:
            new_result[scenario_weather][str(frame_id)] = OrderedDict()
            RA_frame = roi_result[scenario_weather][str(frame_id)].copy()

            if "scenario_go" in RA_frame:
                scenario_go = RA_frame["scenario_go"]
                del RA_frame["scenario_go"]

            if not topk is None:
                # TODO
                if method == "BP":
                    topk_keys = list(
                        dict(sorted(RA_frame.items(), key=lambda x: x[1][1])[::-1]).keys())[:topk]
                    
                    for actor_id in RA_frame:
                        score = RA_frame[actor_id][1]
                        # is_risky = (actor_id in topk_keys) and (score > threshold) and scenario_go < 0.4
                        is_risky = (actor_id in topk_keys)
                        new_result[scenario_weather][str(
                            frame_id)][str(int(actor_id)%65536)] = bool(is_risky)

                elif method == "BCP":
                    topk_keys = list(
                        dict(sorted(RA_frame.items(), key=lambda x: x[1][0])[::-1]).keys())[:topk]

                    for actor_id in RA_frame:
                        score = RA_frame[actor_id][0]
                        is_risky = (actor_id in topk_keys) and (score - scenario_go > threshold and scenario_go < 0.5)
                        # is_risky = (actor_id in topk_keys) and (scenario_go < 0.5)
                        new_result[scenario_weather][str(
                            frame_id)][str(int(actor_id)%65536)] = bool(is_risky)

                elif method in ["DSA", "RRL"]:
                    topk_keys = list(
                        dict(sorted(RA_frame.items(), key=lambda x: x[1])[::-1]).keys())[:topk]

                    for actor_id in RA_frame:
                        score = RA_frame[actor_id]
                        is_risky = (actor_id in topk_keys)
                        # is_risky = (actor_id in topk_keys) and score > threshold
                        new_result[scenario_weather][str(
                            frame_id)][str(int(actor_id)%65536)] = bool(is_risky)

                else:
                    print("Method Error!!!")
                    exit()

            
            else:
                for actor_id in RA_frame:
                    if method == "BP":    # score > threshold == 0.35 and scenario_go < 0.4
                        score = RA_frame[actor_id][1]
                        is_risky = (score > threshold) and scenario_go < 0.4
                    elif method == "BCP":     # score - scenario_go > threshold == 0.2 and scenario_go < 0.5
                        score = RA_frame[actor_id][0]
                        is_risky = (score - scenario_go > threshold and scenario_go < 0.5)
                    else:
                        # threshold: {DSA:0.9, RRL:0.8}
                        score = RA_frame[actor_id]
                        is_risky = (score > threshold)

                    new_result[scenario_weather][str(
                        frame_id)][str(int(actor_id)%65536)] = bool(is_risky)


    return new_result


def main(method):

    for _type in data_type:
        roi_result = read_data(_type, method)
        new_RA = ROI_transpose(roi_result)

        if save_result:
            with open(os.path.join(f"new_{_type}.json"), 'w') as f:
                json.dump(new_RA, f, indent=4)

        print(_type, " Done")
        print("==============================================")


if __name__ == '__main__':
    method = "BCP"
    main(method)
