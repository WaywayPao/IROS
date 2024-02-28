import os
import json
import argparse
import numpy as np

from filter import raw_score_filter as Filter
from transpose import ROI_transpose
from collections import OrderedDict
from sklearn.metrics import average_precision_score


PRED_SEC = 4
FRAME_PER_SEC = 20
OPTION = 3
# TOWN = ['10', 'A6', 'B3']

roi_root = "./model_transpose"
dataset_root = "/media/waywaybao_cs10/DATASET/RiskBench_Dataset"
data_types = ['interactive', 'non-interactive', 'collision', 'obstacle'][:]

Method = ["Random", "Range", "Kalman filter", "Social-GAN",
          "MANTRA", "QCNet", "DSA", "RRL", "BP", "BCP", "pf"][:]


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
        
        # basic = '_'.join(scenario_weather.split('_')[:-3])
        # if not basic in ['10_t2-6_1_c_l_r_1_0', '10_t3-2_0_c_r_l_1_s', '10_t3-3_0_c_r_f_1_0']:
        #     is_del = True
        
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


def cal_critical_point(data_type, basic, variant, variant_path, behavior_dict, risky_id):

    critical_frame = -1

    if data_type in ["interactive", "obstacle"]:

        start, end = behavior_dict[basic][variant]
        actor_data_path = os.path.join(variant_path, "actors_data")
        img_path = os.path.join(variant_path, "rgb/front")

        last_frame = len(os.listdir(img_path))
        if last_frame < end:
            end = last_frame
            print("behavior leak!", data_type, basic, variant, risky_id)

        nearest_frame = -1
        nearest_dis = float('inf')

        for frame in range(start, end+1):
            frame_json = json.load(
                open(os.path.join(actor_data_path, f"{frame:08d}.json")))

            if str(int(risky_id) % 65536) in frame_json:
                risky_id = str(int(risky_id) % 65536)

            elif str(int(risky_id) % 65536+65536) in frame_json:
                risky_id = str(int(risky_id) % 65536+65536)

            dis = frame_json[risky_id]["distance"]

            if dis < nearest_dis:
                nearest_dis = dis
                nearest_frame = frame

        critical_frame = int(nearest_frame)

    elif data_type == "collision":
        critical_frame = int(
            json.load(open(os.path.join(variant_path, "collision_frame.json")))["frame"])

    return critical_frame


def gen_GT_risk(data_type, behavior_dict=None):

    _type_path = os.path.join(dataset_root, data_type)
    risky_dict = OrderedDict()
    critical_dict = OrderedDict()

    for basic in sorted(os.listdir(_type_path)):
        # if not basic[:2] in ["10", "A6", "B3"]:
        #     continue
        basic_path = os.path.join(_type_path, basic, "variant_scenario")

        for variant in sorted(os.listdir(basic_path)):

            variant_path = os.path.join(basic_path, variant)
            interactor_path = os.path.join(
                variant_path, "actor_attribute.json")

            if data_type == "non-interactive":
                risky_dict[basic+"_"+variant] = [None]
                critical_dict[basic+"_"+variant] = [None]

            elif data_type in ['interactive', 'collision']:
                risky_id = json.load(open(interactor_path))["interactor_id"]
                risky_dict[basic+"_"+variant] = [str(int(risky_id) % 65536)]

                critical_dict[basic+"_"+variant] = cal_critical_point(
                    data_type, basic, variant, variant_path, behavior_dict, risky_id)

            else:
                risky_id = int(
                    json.load(open("./GT_risk/obstacle.json"))[basic+"_"+variant][0])
                critical_dict[basic+"_"+variant] = cal_critical_point(
                    data_type, basic, variant, variant_path, behavior_dict, risky_id)

    # with open(f"GT_risk/{data_type}.json", 'w') as f:
    #     json.dump(risky_dict, f, indent=4)

    # with open(f"GT_critical_point/{data_type}.json", 'w') as f:
    #     json.dump(critical_dict, f, indent=4)

    return risky_dict, critical_dict


def cal_confusion_matrix(data_type, roi_result, risky_dict, behavior_dict=None, critical_dict=None):

    TP, FN, FP, TN = 0, 0, 0, 0

    TOTAL_FRAME = FRAME_PER_SEC*PRED_SEC
    f1_sec = np.zeros((PRED_SEC+1, 4))
    cnt = 0
    ambiguous_FN = 0
    ambiguous_FP = 0
    accum_critical_frame = 0
    
    for scenario_weather in roi_result.keys():

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])
        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = -999, 999

        if data_type != "non-interactive":
            critical_frame = critical_dict[scenario_weather]
            # critical_frame = end
        else:
            critical_frame = 999

        # start, end = start+5, end-5
        accum_critical_frame += critical_frame-start
        
        if data_type != "non-interactive":
            risky_id = risky_dict[scenario_weather][0]
        else:
            risky_id = None

        for frame_id in roi_result[scenario_weather]:

            # if (start <= int(frame_id) <= end):
            #     continue
            # if not (start-10 <= int(frame_id) < start or end < int(frame_id) < end+10):
            #     continue

            _TP, _FN, _FP, _TN = 0, 0, 0, 0
            all_actor_id = list(
                roi_result[scenario_weather][str(frame_id)].keys())

            behavior_stop = (start <= int(frame_id) <= end)
            # behavior_stop = True

            for actor_id in all_actor_id:

                is_risky = roi_result[scenario_weather][str(
                    frame_id)][actor_id]

                if behavior_stop and (str(int(actor_id) % 65536) == risky_id
                                      or str(int(actor_id) % 65536+65536) == risky_id):
                    if is_risky:
                        _TP += 1
                    else:
                        _FN += 1
                        if start<=int(frame_id)<start+10 or end-10<int(frame_id)<=end:
                            ambiguous_FN += 1

                else:
                    if is_risky:
                        _FP += 1
                        if start-10<=int(frame_id)<start or end<int(frame_id)<=end+10:
                            ambiguous_FP += 1
                    else:
                        _TN += 1

            if 0 <= critical_frame - int(frame_id) < TOTAL_FRAME:
                f1_sec[(critical_frame - int(frame_id))//FRAME_PER_SEC +
                       1, :] += np.array([_TP, _FN, _FP, _TN])

            TP += _TP
            FN += _FN
            FP += _FP
            TN += _TN

    # print("accum_critical_frame :", accum_critical_frame/len(roi_result))
    print("ambiguous_FN :", ambiguous_FN)
    print("ambiguous_FP :", ambiguous_FP)
    return np.array([TP, FN, FP, TN]).astype(int), f1_sec


def cal_IDsw(data_type, roi_result, risky_dict, behavior_dict):

    IDcnt = 0
    IDsw = 0

    for scenario_weather in roi_result.keys():

        pre_frame_info = None

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])
        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = 0, 999

        for frame_id in roi_result[scenario_weather]:

            cur_frame_info = roi_result[scenario_weather][str(frame_id)]
            all_actor_id = list(cur_frame_info.keys())

            for actor_id in all_actor_id:

                # if actor_id != risky_dict[scenario_weather][0] or not (start <= int(frame_id) <= end):
                #     continue

                IDcnt += 1
                if not pre_frame_info is None:
                    if actor_id in pre_frame_info and cur_frame_info[actor_id] != pre_frame_info[actor_id]:
                        IDsw += 1

            pre_frame_info = cur_frame_info


    return IDcnt, IDsw


def cal_MOTA(cur_confusion_matrix, IDsw, IDcnt):

    FN, FP = cur_confusion_matrix[1:3]
    MOTA = 1-(FN+FP+IDsw)/IDcnt
    

    return MOTA


def cal_PIC(data_type, roi_result, behavior_dict, risky_dict, critical_dict, EPS=1e-8):

    assert data_type != "non-interactive", "non-interactive can not calculate PIC!!!"

    PIC = 0

    for scenario_weather in roi_result.keys():

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])
        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = 0, 999

        end_frame = critical_dict[scenario_weather]
        risky_id = risky_dict[scenario_weather][0]

        for frame_id in roi_result[scenario_weather]:

            if int(frame_id) > int(end_frame):
                break

            all_actor_id = list(roi_result[scenario_weather][frame_id].keys())

            if len(all_actor_id) == 0:
                continue

            TP, FN, FP, TN = 0, 0, 0, 0
            behavior_stop = (start <= int(frame_id) <= end)

            for actor_id in all_actor_id:

                is_risky = roi_result[scenario_weather][frame_id][actor_id]

                if behavior_stop and (str(int(actor_id) % 65536) == risky_id
                                      or str(int(actor_id) % 65536+65536) == risky_id):
                    if is_risky:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if is_risky:
                        FP += 1
                    else:
                        TN += 1

            recall, precision, f1 = compute_f1(
                np.array([TP, FN, FP, TN]).astype(int))

            # exponential F1 loss
            if TP+FP+FN > 0 and 0 <= int(end_frame)-int(frame_id) < 60:
                c = 1.0
                # PIC += -(FP+FN)*(np.exp(c*-(int(end_frame)-int(frame_id))/60)
                #         * np.log(f1 + EPS))
                PIC += -(np.exp(c*-(int(end_frame)-int(frame_id))/60)
                        * np.log(f1 + EPS))


    PIC = PIC/len(roi_result.keys())
    return PIC


def cal_consistency(data_type, roi_result, risky_dict, critical_dict, EPS=1e-05):

    FRAME_PER_SEC = 20
    TOTAL_FRAME = FRAME_PER_SEC*3

    # consistency in 0~3 seconds
    consistency_sec = np.ones(4)*EPS
    consistency_sec_cnt = np.ones(4)*EPS

    for scenario_weather in roi_result.keys():

        end_frame = critical_dict[scenario_weather]
        risky_id = str(risky_dict[scenario_weather][0])
        is_risky = [None]*TOTAL_FRAME

        for frame_id in roi_result[scenario_weather]:

            if int(frame_id) > end_frame:
                break

            if end_frame - int(frame_id) >= TOTAL_FRAME:
                continue

            cur_frame_info = roi_result[scenario_weather][str(frame_id)]
            all_actor_id = list(cur_frame_info.keys())

            if not risky_id in all_actor_id:
                continue

            is_risky[end_frame - int(frame_id)] = cur_frame_info[risky_id]

        for i in range(0, TOTAL_FRAME, FRAME_PER_SEC):

            if np.any(is_risky[i:i+FRAME_PER_SEC] != None):
                consistency_sec_cnt[i//FRAME_PER_SEC+1] += 1

                if not False in is_risky[:i+FRAME_PER_SEC]:
                    consistency_sec[i//FRAME_PER_SEC+1] += 1
                # else:
                #     if i//FRAME_PER_SEC+1 == 1:
                #         print(scenario_weather, end_frame)
            else:
                print(is_risky[i:i+FRAME_PER_SEC])

    return consistency_sec, consistency_sec_cnt


def cal_FA(roi_result):

    FA = 0
    n_frame = 0

    for scenario_weather in roi_result:

        for frame_id in roi_result[scenario_weather]:

            if len(roi_result[scenario_weather][frame_id]) == 0:
                continue

            if True in roi_result[scenario_weather][frame_id].values():
                FA += 1

            n_frame += 1

    return FA, n_frame


def cal_AP(data_type, roi_result, risky_dict, behavior_dict=None, critical_dict=None):
    
    pred_metrics = []
    target_metrics = []

    FRAME_PER_SEC = 20
    TOTAL_FRAME = FRAME_PER_SEC*3

    for scenario_weather in roi_result.keys():

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])

        actor_type = basic.split('_')[3]
        
        # if not actor_type in ["c", "t"]:
        #     continue
        # if not actor_type in ["m", "b"]:
        #     continue
        # if not actor_type in ["p"]:
        #     continue

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
        is_risky = [None]*TOTAL_FRAME


        for frame_id in roi_result[scenario_weather]:

            if int(frame_id) > end_frame:
                break
            if end_frame - int(frame_id) >= TOTAL_FRAME:
                continue

            all_actor_id = list(
                roi_result[scenario_weather][str(frame_id)].keys())

            behavior_stop = (start <= int(frame_id) <= end)

            for actor_id in all_actor_id:

                is_pred_risky = roi_result[scenario_weather][str(
                    frame_id)][actor_id]
                is_gt_risk = int(behavior_stop and (str(int(actor_id) % 65536) == risky_id or str(int(actor_id) % 65536+65536) == risky_id))

                pred_metrics.append(int(is_pred_risky))
                target_metrics.append(int(is_gt_risk))

    AP = average_precision_score(target_metrics, pred_metrics)

    return AP


def compute_f1(confusion_matrix, EPS=1e-5):

    TP, FN, FP, TN = confusion_matrix

    recall = TP / (TP+FN+EPS)
    precision = TP / (TP+FP+EPS)
    f1_score = 2*precision*recall / (precision+recall+EPS)

    return recall, precision, f1_score


def show_result(_type, method, confusion_matrix, AP, IDcnt, IDsw, MOTA, PIC=-1, consistency_sec=-1, consistency_sec_cnt=-1, FA=-1, n_frame=-1, attribute=None, f1_sec=None, scenario_n=1):

    TP, FN, FP, TN = confusion_matrix
    recall, precision, f1_score = compute_f1(confusion_matrix)

    # print(
    #     f"Method: {method}\tAttribute: {attribute}, type: {_type}")
    print(f"TP: {TP},  FN: {FN},  FP: {FP},  TN: {TN}")
    print(
        f"Recall: {recall*100:.2f}%  Precision: {precision*100:.2f}%  F1-Score: {f1_score*100:.2f}%")
    print(f"AP: {AP*100:.2f}%")
    # print(f"N: {int(TP+FN+FP+TN)}")
    # print(f"Accuracy: {(TP+TN)*100/(TP+FN+FP+TN):.2f}%")
    print(f"IDcnt: {IDcnt}, IDsw: {IDsw}, IDsw rate:{IDsw/IDcnt*100:.2f}%")
    print(f"MOTA: {MOTA*100:.2f}%   PIC: {PIC:.1f}")
    print(f"FA rate: {FA/n_frame*100:.2f}%")

    # for i in range(1, 4):
    #     print(
    #         f"Consistency in {i}s: {consistency_sec[i]/consistency_sec_cnt[i]*100:.2f}%")
        # print(
        #     f"Consistency in {i}s: {int(consistency_sec[i])}/{int(consistency_sec_cnt[i])}")

    total_f1_sec = compute_f1(np.sum(f1_sec, axis=0))[2]
    f1_save = {"f1": total_f1_sec, "Current": {}, "Accumulation": {}}

    # confusion matrix
    f1_sec_sum = np.zeros(4)
    _PIC = 0

    for i in range(1, PRED_SEC+1):

        f1_sec_sum += f1_sec[i]
        r, p, f1 = compute_f1(f1_sec_sum)
        f1_save["Accumulation"][i] = f1

        # print(
        #     f"F1 in 1~{i}s: {f1*100:.2f}%,\tRecall: {r*100:.2f}%,\tPrecision: {p*100:.2f}%")

        r, p, f1 = compute_f1(f1_sec[i])
        f1_save["Current"][i] = f1
        print(f"F1 in {i}s: {f1*100:.2f}%,\tRecall: {r*100:.2f}%,\tPrecision: {p*100:.2f}%, {f1_sec[i]}")


    # with open(os.path.join("f1_result", f"{method}_{_type}.json"), 'w') as f:
    #     json.dump(f1_save, f, indent=4)
    
    print()

    result = {"Method": method, "Attribute": attribute, "type": _type, "confusion matrix": confusion_matrix.tolist(),
              "recall": f"{recall:.4f}", "precision": f"{precision:.4f}", "AP":f"{AP:.4f}",
              "accuracy": f"{(TP+TN)/(TP+FN+FP+TN):.4f}", "f1-Score": f"{f1_score:.4f}",
              "IDcnt": f"{IDcnt}", "IDsw": f"{IDsw}", "IDsw rate": f"{IDsw/IDcnt:.4f}",
              "MOTA": f"{MOTA:.4f}", "PIC": f"{PIC:.1f}", "FA": f"{FA/n_frame:.4f}",
              #   "Consistency_1s": f"{consistency_sec[1]/consistency_sec_cnt[1]:.4f}",
              #   "Consistency_2s": f"{consistency_sec[2]/consistency_sec_cnt[2]:.4f}",
              #   "Consistency_3s": f"{consistency_sec[3]/consistency_sec_cnt[3]:.4f}"}
              "Consistency_cnt_1s": f"{consistency_sec_cnt[1]}",
              "Consistency_cnt_2s": f"{consistency_sec_cnt[2]}",
              "Consistency_cnt_3s": f"{consistency_sec_cnt[3]}",
              "Consistency_1s": f"{consistency_sec[1]}",
              "Consistency_2s": f"{consistency_sec[2]}",
              "Consistency_3s": f"{consistency_sec[3]}",
              "f1 in second": f1_save,}
            
    return result, recall, precision, f1_score


def ROI_evaluation(_type, method, roi_result, risky_dict, critical_dict, behavior_dict=None, attribute=None):

    EPS = 1e-05
    PIC = -1
    FA, n_frame = 0, -1
    consistency_sec = np.zeros(4)
    consistency_sec_cnt = np.ones(4)*EPS
    f1_sec = np.ones(4)*EPS

    confusion_matrix, f1_sec = cal_confusion_matrix(
        _type, roi_result, risky_dict, behavior_dict, critical_dict)
    AP = cal_AP(_type, roi_result, risky_dict, behavior_dict, critical_dict)
    IDcnt, IDsw = cal_IDsw(_type, roi_result, risky_dict, behavior_dict)
    MOTA = cal_MOTA(confusion_matrix, IDsw, IDcnt)

    if _type != "non-interactive":
        PIC = cal_PIC(_type, roi_result, behavior_dict,
                      risky_dict, critical_dict)
        consistency_sec, consistency_sec_cnt = cal_consistency(
            _type, roi_result, risky_dict, critical_dict)
    else:
        FA, n_frame = cal_FA(roi_result)

    metric_result, recall, precision, f1_score = show_result(
        _type, method, confusion_matrix, AP, IDcnt, IDsw, MOTA, PIC, consistency_sec, consistency_sec_cnt, FA, n_frame, attribute, f1_sec, scenario_n = len(roi_result.keys()))

    with open(os.path.join("./result_network-pi", f"thres={args.threshold}.json"), 'w') as f:
        json.dump(metric_result, f, indent=4)
    
    if args.save_result:
        save_folder = f"./result/{method}"
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        with open(os.path.join(save_folder, f"{_type}_attr={attribute}_result.json"), 'w') as f:
            json.dump(metric_result, f, indent=4)


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
                    
        # risky_dict, critical_dict = gen_GT_risk(_type, behavior_dict)
        # continue

        for method in Method:

            for attr in attributes:

                roi_result = read_data(
                    _type, method, attr=attr, filter_type=args.filter)

                if len(roi_result) == 0:
                    continue

                if args.transpose:
                    roi_result = ROI_transpose(method,
                                               roi_result, args.threshold, args.topk)

                ROI_evaluation(_type, method, roi_result,
                               risky_dict, critical_dict, behavior_dict, attribute=attr)
                print("#"*40)

        print()
