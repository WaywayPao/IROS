import numpy as np
import os
import json
import math
from statistics import median, mean
from collections import OrderedDict


def bayesian(p_list, p_0=0.7, EPS=1e-7):

    def sigmoid(z):
        return 1.0/(1+np.exp(-z))

    def cal_L(p):
        l = np.log(p/(1-p+EPS)+EPS)
        return l

    def cal_P(l):
        p = 1/(1+np.exp(-l+EPS))
        return p

    def cal_L_t(p_list, t):

        if t == 0:
            return l_0
        L_t = cal_L_t(p_list[:t], t-1) + cal_L(p_list[t])

        return L_t

    # prior
    l_0 = cal_L(p_0)
    t = len(p_list)
    p_list.insert(0, None)

    L_t = cal_L_t(p_list, t)
    P_t = cal_P(L_t)

    return P_t


def raw_score_filter(RA, filter_type="median", sliding_size=5, method="BCP"):

    if filter_type == "median":
        Filter = median
    elif filter_type == "mean":
        Filter = mean
    elif filter_type == "bayesian":
        Filter = bayesian
    else:  # filter_type == "none":
        def Filter(score_list): return score_list[-1]

    RA_new = {}

    for scenario_weather in RA.keys():

        new_scenario_dict = OrderedDict()

        for frame_id in RA[scenario_weather]:
            src_score_dict = RA[scenario_weather][str(frame_id)]
            new_scenario_dict[frame_id] = src_score_dict.copy()

        first_frame_id = int(list(new_scenario_dict.keys())[sliding_size-1])
        last_frame_id = int(list(new_scenario_dict.keys())[-1])

        for frame_id in range(first_frame_id, last_frame_id+1):

            src_score_dict = new_scenario_dict[str(frame_id)]
            new_score_dict = dict()

            for actor_id in src_score_dict:

                if actor_id == "scenario_go":
                    new_score_dict[actor_id] = new_scenario_dict[str(
                        frame_id)][actor_id]
                    continue

                score_list = []
                attn_list = []

                for i in range(frame_id-sliding_size+1, frame_id+1):
                    if actor_id in new_scenario_dict[str(i)]:
                        if method in ["BP", "BCP"]:
                            score_list.append(
                                new_scenario_dict[str(i)][actor_id][0])
                            attn_list.append(
                                new_scenario_dict[str(i)][actor_id][1])
                        else:
                            score_list.append(
                                new_scenario_dict[str(i)][actor_id])

                if method in ["BP", "BCP"]:
                    new_score_dict[actor_id] = [
                        Filter(score_list), Filter(attn_list)]
                else:
                    new_score_dict[actor_id] = Filter(score_list)

            new_scenario_dict[str(frame_id)] = new_score_dict

        RA_new[scenario_weather] = new_scenario_dict

    return RA_new
