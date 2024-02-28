import json
import os
from collections import OrderedDict

METHOD = "qcnet"
ref_path = "./Range"
data_type = ["interactive", "collision", "obstacle", "non-interactive"][:]
A = f"{2.5}A"

def read_json(file_name, is_ref=False):

    json_file = open(file_name)
    data = json.load(json_file)
    json_file.close()

    # if is_ref:
    #     for scenario_weather in data:
    #         for frame in data[scenario_weather]:
    #             for id in data[scenario_weather][frame]:
    #                 data[scenario_weather][frame] = False
    return data


for _type in data_type:

    ref_RA = read_json(os.path.join(ref_path, f"{_type}.json"), is_ref=True)
    
    
    file = f"area_json/{METHOD}_1.5s_test/{A}/{METHOD}_1.5s_{_type}_test_{A}.json"
    src_RA = read_json(file)


    new_RA = OrderedDict()

    for scenario_weather in ref_RA:

        if not scenario_weather in src_RA:
            print(scenario_weather)
            continue

        new_RA[scenario_weather] = OrderedDict()

        for frame in src_RA[scenario_weather]:  # type(frame) == str

            if not frame in ref_RA[scenario_weather]:
                print(scenario_weather, frame)
                continue

            new_RA[scenario_weather][frame] = OrderedDict()

            for id in ref_RA[scenario_weather][frame]:
                new_RA[scenario_weather][frame][id] = False

            if "True" in src_RA[scenario_weather][frame]:
                for id in src_RA[scenario_weather][frame]["True"]:
                    if id in new_RA[scenario_weather][frame]:
                        new_RA[scenario_weather][frame][id] = True

    if not os.path.isdir(f"{METHOD}/{A}"):
        os.makedirs(f"{METHOD}/{A}")

    # with open(f"{METHOD}/{A}/{_type}.json", 'w') as f:
    #     json.dump(new_RA, f, indent=4)
