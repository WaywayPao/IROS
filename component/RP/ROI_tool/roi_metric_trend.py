import os
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from filter import raw_score_filter as Filter
from collections import OrderedDict

PRED_SEC = 3
FRAME_PER_SEC = 20
OPTION = 3

roi_root = "./model_transpose"
dataset_root = "/media/waywaybao_cs10/DATASET/RiskBench_Dataset"
data_types = ['interactive', 'non-interactive', 'collision', 'obstacle'][:]

Method = ["Random", "Range", "Kalman filter", "Social-GAN",
          "MANTRA", "QCNet", "DSA", "RRL", "BP", "BCP", "pf"][:]

TOTAL_FRAME = FRAME_PER_SEC*PRED_SEC


attributes = [["Night", "Rain", "low", "mid", "high"],
              ["_i", "_t", "_s"],
              ["c", "t", "m", "b", "p", "4", "2"],
              [""]][OPTION]



def read_data(data_type, method, attr="", filter_type="none"):

    json_path = os.path.join(roi_root, method, f"{data_type}.json")

    roi_file = open(json_path)
    roi_result = json.load(roi_file)
    roi_file.close()

    # filter scenario
    all_scnarios = list(roi_result.keys())
    for scenario_weather in all_scnarios:

        is_del = False
        
        if OPTION == 0:
            if not attr in scenario_weather:
                is_del = True
        elif OPTION == 1:
            if not attr == scenario_weather[2:4]:
                is_del = True
            
        elif OPTION == 2:
            actor_type = scenario_weather.split('_')[3]
            if attr == "4":
                if not actor_type in ["c", "t"]:
                    is_del = True
            elif attr == "2":
                if not actor_type in ["m", "b"]:
                    is_del = True
            else:
                if attr != actor_type:
                    is_del = True

        if is_del:
            del roi_result[scenario_weather]

    if filter_type != "none":
        roi_result = Filter(roi_result, filter_type=filter_type, method=method)

    return roi_result



def plot_v1(FDE_trend_sec):

    plt.clf()
    x_list = range(0, TOTAL_FRAME)    
    freq = 2    # frequence of x ticks

    plt.plot(x_list[::freq], FDE_trend_sec[::freq], color='r', markersize="2",
                marker="o", label="AVG_FDE")

    plt.xticks(x_list[::freq], rotation=-45, fontsize=8)
    # plt.xlim(0, 11)
    # plt.ylim(lower, upper)
    plt.grid(alpha=0.2)

    plt.xlabel('Frame', fontsize="10")
    plt.ylabel(f"L2 Distance", fontsize="10")
    plt.title(f'Average Max-FDE from Critical Point')
    plt.legend(loc ="upper right")

    plt.savefig(f'./avg_FDE.png', dpi=300)
    # plt.show()
    

def plot(FDE_trend_sec):


    x_list = range(0, TOTAL_FRAME)    
    freq = 1    # frequence of x ticks

    plt.clf()
    fig = plt.figure(figsize=(8,5))

    # fig.set_figwidth(10)
    plt.plot(x_list[::freq], FDE_trend_sec[::freq][::-1], color='r', label="AVG_FDE")

    # plt.xticks(x_list[::freq], rotation=-45, fontsize=8)
    plt.xticks([])
    plt.yticks(np.arange(2.5, 7.5, 0.5))
    # plt.xlim(0, 11)
    # plt.ylim(2.5, 7.5)
    # plt.grid(alpha=0.2)

    # plt.xlabel('Frame', fontsize="10")
    plt.ylabel(f"L2 Distance", fontsize="10")
    plt.title(f'Average Max-FDE from Critical Point')
    plt.legend(loc ="upper left")

    cax = fig.add_axes([0.15, 0.07, 0.72, 0.02])
    cmap = mpl.cm.get_cmap('rainbow')
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, orientation='horizontal', ticks=[0, 0.5, 1])
    cbar.ax.set_xticklabels(["Investigative", "Reactive", "Critical"])
    plt.savefig(f'./avg_FDE.png', dpi=300, bbox_inches='tight')
    # plt.show()
    

def cal_FDE_trend(data_type, roi_result, risky_dict, behavior_dict=None, critical_dict=None):

    FDE_trend_sec = np.zeros((TOTAL_FRAME, 1))
    FDE_trend_cnt = np.zeros((TOTAL_FRAME, 1))


    for scenario_weather in roi_result.keys():

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])
        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = -999, 999

        if data_type != "non-interactive":
            critical_frame = critical_dict[scenario_weather]
        else:
            critical_frame = 999


        for frame_id in roi_result[scenario_weather]:

            all_actor_id = list(
                roi_result[scenario_weather][str(frame_id)].keys())

            FDE_dict = OrderedDict()
            for actor_id in all_actor_id:
                if actor_id == "0":
                    continue
                FDE = roi_result[scenario_weather][str(frame_id)][actor_id]
                FDE_dict[actor_id] = FDE

            if len(FDE_dict) == 0:
                continue

            sorted_keys = list(
                dict(sorted(FDE_dict.items(), key=lambda x: x[1])[::-1]).keys())
            
            if 0 <= critical_frame - int(frame_id) < TOTAL_FRAME:
                FDE_trend_sec[(critical_frame - int(frame_id))] += FDE_dict[sorted_keys[0]]
                FDE_trend_cnt[(critical_frame - int(frame_id))] += 1

    for i in range(TOTAL_FRAME):
        FDE_trend_sec[i] = FDE_trend_sec[i]/FDE_trend_cnt[i]

    return FDE_trend_sec


def cal_AP(data_type, roi_result, risky_dict, behavior_dict=None, critical_dict=None):
    from sklearn.metrics import average_precision_score

    pred_metrics = []
    target_metrics = []


    for scenario_weather in roi_result.keys():

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])

        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = -999, 999

        if data_type != "non-interactive":
            risky_id = risky_dict[scenario_weather][0]
        else:
            risky_id = None
            
        end_frame = critical_dict[scenario_weather]
        risky_id = str(risky_dict[scenario_weather][0])

        for frame_id in roi_result[scenario_weather]:

            if int(frame_id) > end_frame:
                break
            # if end_frame - int(frame_id) >= TOTAL_FRAME:
            #     continue

            all_actor_id = list(
                roi_result[scenario_weather][str(frame_id)].keys())

            behavior_stop = (start <= int(frame_id) <= end)

            for actor_id in all_actor_id:

                pred_risk = min(max(roi_result[scenario_weather][str(frame_id)][actor_id], 1.0), 5.0) / 5.0
                gt_risk = int(behavior_stop and (str(int(actor_id) % 65536) == risky_id or str(int(actor_id) % 65536+65536) == risky_id))

                pred_metrics.append(int(pred_risk))
                target_metrics.append(int(gt_risk))

    AP = average_precision_score(target_metrics, pred_metrics)

    return AP



def ROI_evaluation(_type, method, roi_result, risky_dict, critical_dict, behavior_dict=None, attribute=None):

    FDE_trend_sec = cal_FDE_trend(_type, roi_result, risky_dict, behavior_dict, critical_dict)
    AP = cal_AP(_type, roi_result, risky_dict, behavior_dict, critical_dict)
    print(f"AP: {AP*100:.2f}%")
    plot(FDE_trend_sec)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default="all", required=True, type=str)
    parser.add_argument('--data_type', default='all', required=True, type=str)
    parser.add_argument('--transpose', action='store_true', default=False)
    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--filter', default="none", type=str)
    parser.add_argument('--topk', default=None, type=int)
    parser.add_argument('--save_result', action='store_true', default=False)

    args = parser.parse_args()

    if args.method != 'all':
        Method = [args.method]

    if args.data_type != 'all':
        data_types = [args.data_type]

    for _type in data_types:

        risky_dict = json.load(open(f"./metadata/GT_risk/{_type}.json"))
        critical_dict = json.load(
            open(f"./metadata/GT_critical_point/{_type}.json"))
        behavior_dict = None

        if _type in ["interactive", "obstacle"]:
            behavior_dict = json.load(
                open(f"./metadata/behavior/{_type}.json"))
                    
        for method in Method:

            for attr in attributes:

                roi_result = read_data(
                    _type, method, attr=attr, filter_type=args.filter)

                if len(roi_result) == 0:
                    continue

                ROI_evaluation(_type, method, roi_result,
                               risky_dict, critical_dict, behavior_dict, attribute=attr)
                print("#"*40)

        print()
