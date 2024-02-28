import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict

result_root = "./result_network-pi"


def plot(x_list, data_list, upper, lower, label=[""], unit="", color=['r', 'b', 'g', 'y', 'skyblue'], title=""):
    
    plt.clf()
    freq = 1    # frequence of x ticks

    for i in range(len(data_list)):
        plt.plot(x_list[::freq], data_list[i][::freq], color=color[i], markersize="2",
                    marker="o", label=label[i])

    plt.xticks(x_list[::freq], rotation=-45, fontsize=8)
    # plt.xlim(0, 11)
    plt.ylim(lower, upper)
    plt.grid(alpha=0.2)

    plt.xlabel('Threshold', fontsize="10")
    plt.ylabel(f"{unit}", fontsize="10")
    plt.title(f'{title} in different thresholds')
    plt.legend(loc ="upper right")

    plt.savefig(f'./pi_FDE_{title}.png', dpi=300)
    # plt.show()


def main():

    f1_list = list()
    recall_list = list()
    precision_list = list()
    PIC_list = list()
    MOTA_list = list()
    IDsw_rate_list = list()
    f1_in_1sec_list = list()
    f1_in_2sec_list = list()
    f1_in_3sec_list = list()
    f1_in_4sec_list = list()
    PIC_scale = 704.483678

    thres_list = np.arange(4.0, 36.5, 1.0)
    # thres_list = np.arange(2.0, 9.0, 1)

    for thres in thres_list:
        
        result_json = json.load(open(os.path.join(result_root, f"thres={thres}.json")))
        f1 = float(result_json["f1-Score"])*100
        recall = float(result_json["recall"])*100
        precision = float(result_json["precision"])*100
        PIC = float(result_json["PIC"])/PIC_scale
        MOTA = float(result_json["MOTA"])*100
        IDsw_rate = float(result_json["IDsw rate"])*100
        f1_in_1sec = float(result_json["f1 in second"]["Current"]["1"])*100
        f1_in_2sec = float(result_json["f1 in second"]["Current"]["2"])*100
        f1_in_3sec = float(result_json["f1 in second"]["Current"]["3"])*100
        f1_in_4sec = float(result_json["f1 in second"]["Current"]["4"])*100
        
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        PIC_list.append(PIC)
        MOTA_list.append(MOTA)
        IDsw_rate_list.append(IDsw_rate)
        f1_in_1sec_list.append(f1_in_1sec)
        f1_in_2sec_list.append(f1_in_2sec)
        f1_in_3sec_list.append(f1_in_3sec)
        f1_in_4sec_list.append(f1_in_4sec)

    plot(thres_list, [f1_list, recall_list, precision_list], upper=70.0, lower = 40.0, label=["F1", "Recall", "Precision"], unit="%", title="F1")
    plot(thres_list, [PIC_list], upper=0.280, lower = 0.230, label=["PIC"], unit="(Normalization)", title="PIC")
    plot(thres_list, [MOTA_list], upper=93.0, lower = 87.0, label=["MOTA"], unit="%", title="MOTA")
    plot(thres_list, [IDsw_rate_list], upper=2.0, lower = 0.6, label=["IDsw rate"], unit="%", title="IDsw_rate")
    plot(thres_list, [f1_in_1sec_list, f1_in_2sec_list, f1_in_3sec_list, f1_in_4sec_list], upper=70.0, lower = 5.0, label=["F1_1s", "F1_2s", "F1_3s", "F1_4s"], title="F1_in_sec")



if __name__ == '__main__':

    main()
