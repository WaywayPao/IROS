import argparse
import json
import copy
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from TP_model.model import TP_MODEL
from BCP.models import GCN as PF_BCP
from OIECR.gen_waypoints  import gen_waypoint
from OIECR.cal_FDE  import cal_importance_score
from gen_PF import Render_PF as Render_PF, get_seg_mask

from tqdm import tqdm
import torch
import torch.nn as nn


IMG_H = 100
IMG_W = 200

data_types = ['interactive', 'non-interactive', 'collision', 'obstacle'][:1]
data_type = data_types[0]


def to_device(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device)

    return x


def count_parameters(model):

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total parameters: ', params)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)


def load_weight(model, checkpoint):
    model = torch.load(checkpoint)
    return copy.deepcopy(model)


def create_TP_model(tp_path, device):

    model = TP_MODEL().to(device)
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(tp_path))

    # model = load_weight(model, tp_path)

    count_parameters(model)
    model = model.to(device)

    return model
 

def create_BCP_model(args, device):

    assert args.ckpt_path != ""
    
    model = PF_BCP("pf", args.time_step, pretrained=args.pretrained,
                  partialConv=args.partial_conv, use_target_point=args.use_target_point, NUM_BOX=args.num_box)

    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.ckpt_path))

    # model = load_weight(model, args.ckpt_path)
    # if not isinstance(model, nn.DataParallel):
    #     model = nn.DataParallel(model)

    count_parameters(model)
    model = model.to(device)

    return model


def read_scenario(data_root):
    scenario_list = []
    
    type_path = os.path.join(data_root, "interactive")
    for basic in os.listdir(type_path):
        if not basic[:2] in ["10", "A6", "B3"]:
            continue
        basic_path = os.path.join(type_path, basic, "variant_scenario")
        for variant in os.listdir(basic_path):
            variant_path = os.path.join(basic_path, variant)
            scenario_path = os.path.join(basic_path, variant, "cvt_bev-seg")
            scenario_list.append(scenario_path)

    return scenario_list[:]


def load_scenario(args, scenario_path):

    scenario_list = []
    basic, _, variant, _, frame = scenario_path.split('/')[-5:]
    data_root = os.path.join(args.data_root, data_type)

    variant_path = os.path.join(data_root, basic, "variant_scenario", variant)
    frame_id = int(frame.split('.')[0])
    
    for t in range(frame_id-args.time_step+1, frame_id+1):
        if args.use_gt:
            seg_path = os.path.join(variant_path, "bev-seg", f"{t:08d}.npy")
            raw_bev_seg = np.load(seg_path)
        else:
            seg_path = os.path.join(variant_path, "cvt_bev-seg", f"{t:08d}.npy")        
            raw_bev_seg = np.load(seg_path)

        bev_seg = get_seg_mask(raw_bev_seg[:, :100], args.use_gt)
        scenario_list.append(bev_seg)

    scenario_list = torch.stack(scenario_list).unsqueeze(0).cuda(0)
    return scenario_list


def test_BCP(model, PF_list, target_point, device):
    
    roi_dict = OrderedDict()
    actor_id_list = []
    target_point = target_point.unsqueeze(0)

    for actor_id in PF_list[-1].keys():
        if not actor_id in ["attractive", "roadline"]:
            actor_id_list.append(actor_id)

    for actor_id in actor_id_list:
        gt_pf_list = list()

        for pf in PF_list:
            if actor_id in pf:
                actor_pf = pf[actor_id]
            else:
                actor_pf = torch.zeros((100,200)).cuda(0)

            roadline_pf = pf['roadline']
            attractive_pf = pf['attractive']
 
            if actor_id != "all_actor":
                gt_pf = ((pf["all_actor"]-actor_pf)+roadline_pf+attractive_pf).clip(0.1, 90)
            else:
                gt_pf = (actor_pf+roadline_pf+attractive_pf).clip(0.1, 90)

            gt_pf = gt_pf.unsqueeze(0).float()
            gt_pf_list.append(gt_pf)

        seg_inputs = torch.stack(gt_pf_list).unsqueeze(0)
        trackers = torch.zeros([1, args.time_step, args.num_box, 4]).cuda(0)
        mask = torch.ones((1, args.time_step, 4, IMG_H, IMG_W)).cuda(0)        
        
        pred = model(seg_inputs, target_point, trackers, mask, device)
        score = pred.detach().to('cpu').numpy().reshape((-1, 1))[0].item()

        roi_dict[actor_id] = score

    scenario_go = 1-roi_dict["all_actor"]

    for actor_id in roi_dict:
        if actor_id == "all_actor":
            continue
        
        raw_score = roi_dict[actor_id]
        score_go = (1-raw_score)
        score = score_go-scenario_go

        is_risky = (score>scenario_go/2 and scenario_go<0.5) or \
                (score>0.18 and 0.75>scenario_go>0.5) or \
                (score>0.12 and scenario_go>0.75)
        
        roi_dict[actor_id] = is_risky

    return roi_dict


def test_OIECR(target_point, potential_field, method):

    roi_dict = OrderedDict()
    gy, gx = target_point.tolist()

    all_actor_pf = potential_field["all_actor"]
    roadline_pf = potential_field["roadline"]
    attractive_pf = potential_field["attractive"]

    all_actor_wp = gen_waypoint(gx=gx, gy=gy, potential_map=all_actor_pf, res=1.0)

    for actor_id in potential_field:
        if actor_id in ["all_actor", "roadline", "attractive"]:
            continue

        remove_pf = ((all_actor_pf-potential_field[actor_id])+roadline_pf+attractive_pf).clip(0.1, 90)
        actor_wp = gen_waypoint(gx=gx, gy=gy, potential_map=remove_pf, res=1.0)
        
        if method == "PF-ADE":
            raw_score = cal_importance_score(all_actor_wp, actor_wp, k=20)
            score = raw_score / 330.4646170881637
            is_risky = (score > 0.1)
        
        elif method == "PF-FDE":
            raw_score = cal_importance_score(all_actor_wp[-1:], actor_wp[-1:], k=1)
            score = raw_score / 34.132096331752024
            is_risky = (score > 0.085)

        roi_dict[actor_id] = is_risky

    return roi_dict


torch.no_grad()
def demo(bev_seg_list, TP_model, model=None):

    st = time.time()
    pred_x, pred_y = TP_model(bev_seg_list)

    if args.method == "PF-BCP":
        PF_list, target_point = Render_PF(pred_x[0], pred_y[0], bev_seg_list[0])
        roi_dict = test_BCP(model, PF_list,  target_point, device)
    
    elif args.method in ["PF-FDE", "PF-ADE"]:
        PF_list, target_point = Render_PF(pred_x[0], pred_y[0], bev_seg_list[0,-1:])
        roi_dict = test_OIECR(target_point, PF_list[-1], method=args.method)

    else:
        print("Error method")
        exit()
    ed = time.time()

    # for actor_id in roi_dict:
    #     print(actor_id, f": {roi_dict[actor_id]}")

    return ed-st


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', default=f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset", type=str)
    parser.add_argument('--data_root', default=f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data", type=str)
    parser.add_argument('--method', choices=["PF-BCP", "PF-FDE", "PF-ADE"], type=str, required=True)
    parser.add_argument('--ckpt_path', default="", type=str)
    parser.add_argument('--tp_path', default=f"./tp.pth", type=str)
    parser.add_argument('--scenario_path', type=str, required=False)
    parser.add_argument('--use_gt', action='store_true', default=False)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)

    parser.add_argument('--time_step', default=5, type=int)
    parser.add_argument('--num_box', default=30, type=int)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--partial_conv', default=True, type=bool)
    parser.add_argument('--use_target_point', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Device:", device)
    
    N = 1
    total_time = 0
    frame_cnt = 0
    scenario_list = read_scenario(args.data_root)
    TP_model = create_TP_model(args.tp_path, device)
    TP_model.eval()

    if args.method == "PF-BCP":
        downstream_model = create_BCP_model(args, device)
        downstream_model.eval()
    else:
        downstream_model = None

    start = time.time()
    for idx, scenario in enumerate(sorted(scenario_list), 1):
        scenario_time = 0
        for frame_npy in sorted(os.listdir(scenario))[args.time_step:]:
            npy_path = os.path.join(scenario, frame_npy)
        
            ## rgb front-view image
            # bev_seg_list = load_scenario(args)

            ## bev segmentation image
            bev_seg_list = load_scenario(args, npy_path)

            for _ in range(N):
                time_elapsed = demo(bev_seg_list.clone(), TP_model, downstream_model)
                scenario_time += time_elapsed
                frame_cnt += 1

        print(f"{args.method} {idx:3d}/{len(scenario_list)}\tInference Time: {scenario_time:4.6f}s\t \
              AVG Time: {scenario_time/(len(os.listdir(scenario))-args.time_step+1):4.6f}")
        total_time += scenario_time

    end = time.time()

    print("="*30)
    print(f"{args.method} Avg inference Time: {total_time/frame_cnt:4.6f}s")
    print(f"{args.method} Overall Time: {end-start:4.6f}s")




