import argparse
import json
import copy
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from models import GCN as Model

from tqdm import tqdm
import torch
import torch.nn as nn
data_types = ['interactive', 'non-interactive', 'collision', 'obstacle'][:1]
# train_town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"] # 1350, (45, 30)
# val_town = ["5_"]
test_town = ["10", "A6", "B3"]   # 515, (47, 11)
town = test_town



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


def create_model(args, device):

    model = Model(args.method, args.time_step, pretrained=args.pretrained,
                  partialConv=args.partial_conv, use_target_point=args.use_target_point, NUM_BOX=args.num_box)

    if args.ckpt_path != "":
        model = load_weight(model, args.ckpt_path)
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    count_parameters(model)
    model = model.to(device)

    return model


def read_target_point():
    target_point_dict = OrderedDict()
    
    for _type in data_types:
        target_point_dict[_type] = json.load(open(f"../TP_model/tp_prediction/{_type}_2024-2-29_232906.json"))

    return target_point_dict


def read_scenario():

    scenario_list = []
    cnt = 0

    for _type in data_types:
       
        data_root = os.path.join(args.sample_root, _type)
        box_3d_root = os.path.join(args.sample_root, _type)

        for basic in sorted(os.listdir(data_root)):
            if not basic[:2] in town:
                continue
            basic_path = os.path.join(data_root, basic, "variant_scenario")

            for variant in sorted(os.listdir(basic_path)):
                variant_path = os.path.join(data_root, basic, "variant_scenario", variant)
                tracking_path = os.path.join(variant_path, "tracking.npy")
                tracking_npy = np.load(tracking_path)

                frame_id = 0
                for sample in tracking_npy:
                    if sample[0] < args.time_step:
                        continue

                    if frame_id != sample[0]:
                        frame_id = str(sample[0])
                        scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+"all_actor")

                    actor_id = str(sample[1])
                    scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+actor_id)
                    cnt += 1


                # bev_seg_path = os.path.join(variant_path, "cvt_bev-seg")
                
                # bev_box = json.load(open(os.path.join(box_3d_root, basic, "variant_scenario", variant, "bev_box.json")))

                # for seg_frame in sorted(os.listdir(bev_seg_path))[4:]:
                #     frame_id = int(seg_frame.split('.')[0])
                #     frame_box = bev_box[f"{frame_id:08d}"]
                #     scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+'all_actor')

                #     for actor_id in frame_box:
                #         scenario_list.append(_type+'#'+basic+'#'+variant+'#'+str(frame_id)+'#'+actor_id)
                #         cnt += 1

    print(cnt)

    return scenario_list


def save_roi_json(roi_dict, json_name):

    new_tp_dict = OrderedDict()

    for scenario, pred_stop in roi_dict.items():
        data_type, basic, variant, frame_id, actor_id = scenario.split('#')

        if not data_type in new_tp_dict:
            new_tp_dict[data_type] = OrderedDict()
        if not basic+'_'+variant in new_tp_dict[data_type]:
            new_tp_dict[data_type][basic+'_'+variant] = OrderedDict()
        if not f"{int(frame_id)}" in new_tp_dict[data_type][basic+'_'+variant]:
            new_tp_dict[data_type][basic+'_'+variant][f"{int(frame_id)}"] = OrderedDict()
        
        new_tp_dict[data_type][basic+'_'+variant][f"{int(frame_id)}"][actor_id] = pred_stop

    for data_type in new_tp_dict:
        with open(f"./ROI/{data_type}_{json_name}.json", "w") as f:
            json.dump(new_tp_dict[data_type], f, indent=4)


def test(args, model, scenario_list, target_point_dict, device):

    roi_dict = OrderedDict()
    model.eval()

    torch.no_grad()
    with tqdm(scenario_list, desc='Testing', unit='sample') as pbar:
        # pbar.set_description(f"Testing")
        for sample in pbar:
            data_type, basic, variant, frame, actor_id = sample.split('#')

            variant_path = os.path.join(args.data_root, data_type, basic, "variant_scenario", variant)

            gt_pf_list = []
            frame = int(frame)
            for frame_id in range(frame-args.time_step+1, frame+1):

                if args.use_gt:
                    pf_path = os.path.join(variant_path, "actor_pf_npy", f"{frame_id:08d}.npy")
                else:
                    pf_path = os.path.join(variant_path, "pre_cvt_clus_actor_pf_npy", f"{frame_id:08d}.npy")

                npy_file = np.load(pf_path, allow_pickle=True).item()

                if actor_id in npy_file:
                    actor_pf = npy_file[actor_id]
                else:
                    actor_pf = np.zeros((100,200))

                roadline_pf = npy_file['roadline']
                attractive_pf = npy_file['attractive']
                # attractive_pf = np.zeros((100,200))

                if actor_id != "all_actor":
                    gt_pf = ((npy_file["all_actor"]-actor_pf)+roadline_pf+attractive_pf).clip(0.1, 90)
                else:
                    gt_pf = (actor_pf+roadline_pf+attractive_pf).clip(0.1, 90)

                gt_pf = np.expand_dims(gt_pf, 0).astype(np.float32)
                gt_pf_list.append(torch.from_numpy(gt_pf))

            gt_pf_list = torch.stack(gt_pf_list).unsqueeze(0)

            # BxTxCxHxW
            seg_inputs = to_device(gt_pf_list, device)

            # Bx2
            target_point = target_point_dict[data_type][basic+'_'+variant][f"{frame:08d}"]
            target_point = torch.Tensor(target_point).unsqueeze(0)
            target_point = to_device(target_point, device)

            # BxTxNx4
            trackers = np.zeros([1, args.time_step, args.num_box, 4]).astype(np.float32)
            trackers = to_device(trackers, device)
            # BxTxCxHxW
            mask = torch.ones((1, args.time_step, 4, 100, 200))
            mask = to_device(mask, device)
            
            pred = model(seg_inputs, target_point, trackers, mask, device)
            roi_dict[sample] = pred.detach().to('cpu').numpy().reshape((-1, 1))[0].item()

            pbar.set_postfix(score=pred[0].item())

    return roi_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_root', default=f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset", type=str)
    parser.add_argument('--data_root', default=f"/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data", type=str)
    parser.add_argument('--method', choices=["vision", "bev_seg", "pf"], type=str, required=True)
    parser.add_argument('--ckpt_path', default="", type=str)
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
    print("Device:", device)

    target_point_dict = read_target_point()
    scenario_list = read_scenario()
    
    model = create_model(args, device)

    roi_dict = test(args, model, scenario_list[:],  target_point_dict, device)
    save_roi_json(roi_dict, json_name=args.ckpt_path.split('/')[-2])





