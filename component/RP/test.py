import time
import os
import copy
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

from config import parse_args
from models import MODEL
from dataset import PotentialFieldDataset


def load_weight(model, checkpoint):
    
    model = torch.load(checkpoint)
    return copy.deepcopy(model)


def count_parameters(model):

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total parameters: ', params)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)


def create_model(args, device):

    assert args.phase == 'train' or args.ckpt_path != "", "No ckpt_path!!!"

    model = MODEL().to(device)
    if args.ckpt_path != "":
        model = load_weight(model, args.ckpt_path)
    model = nn.DataParallel(model).to(device)
    count_parameters(model)

    return model
    

def rearange_dict(src_dict, town=["10", "A6", "B3"]):
    
    final_reachable_points = json.load(open("./utils/final_reachable_points.json"))
    new_dict = OrderedDict()

    for basic in final_reachable_points:
        if not basic[:2] in town:
            continue

        new_dict[basic] = OrderedDict()
        
        for variant in final_reachable_points[basic]:
            new_dict[basic][variant] = OrderedDict()

            for frame_id in final_reachable_points[basic][variant]:
                new_dict[basic][variant][frame_id] = OrderedDict()

                for actor_id in final_reachable_points[basic][variant][frame_id]:
                    x, y = src_dict[basic+'#'+variant+'#'+frame_id+'#'+actor_id]
                    new_dict[basic][variant][frame_id][actor_id] = [x, y]

    return new_dict


def test(args, device, test_loader, model, criterion):

    reachable_point_dict = OrderedDict()
    model.eval()

    total_infer_time = 00
    total_load_time = 0.
    running_loss = 0.0
    start = time.time()
    infer_end = time.time()
    
    with tqdm(test_loader, unit="batch") as tepoch:

        for rgb_inputs, trajs, data_attr in tepoch:
            tepoch.set_description(f"Epoch 1/1")

            rgb_inputs = rgb_inputs.to(device, dtype=torch.float32)
            # trajs = trajs.to(device, dtype=torch.float32).reshape(-1, 1, 160, 80)
            trajs = trajs.to(device, dtype=torch.float32)
            basic, variant, frame_id, actor_id = data_attr
            B = rgb_inputs.shape[0]

            infer_start = time.time()
            load_time = infer_start - infer_end
            total_load_time += load_time
            
            outputs = model(rgb_inputs)
            infer_end = time.time()
            infer_time = infer_end-infer_start
            total_infer_time += infer_time
            
            loss = criterion(outputs, trajs)


            # trajs[:, 0] = trajs[:, 0]*160
            # trajs[:, 1] = trajs[:, 1]*80
            # outputs[:, 0] = outputs[:, 0]*160
            # outputs[:, 1] = outputs[:, 1]*80
            # print("trajs:", torch.tensor(trajs, dtype=int))
            # print("outputs:", torch.tensor(outputs, dtype=int))
            # print("#"*20)

            # statistics
            running_loss += loss.item()*B
            tepoch.set_postfix(loss=loss.item())

            for b in range(B):
                keys = basic[b]+'#'+variant[b]+'#'+frame_id[b]+'#'+actor_id[b]
                x, y = outputs[b].tolist()
                # reachable_point_dict[keys] = [x*160., y*80.]
                reachable_point_dict[keys] = [x, y]


    test_loss = running_loss/(len(test_loader.dataset))
    elapsed = time.time() - start

    print(f"Testing Loss: {test_loss:.4f}")
    print(f"Testing complete in {int(elapsed//60):4d}m {int(elapsed)%60:2d}s")
    print(f"Total inference time in batch_size={args.batch_size}: {total_infer_time:.8f}s")
    print(f"Total loading time in batch_size={args.batch_size}: {total_load_time:.8f}s")
    print(f"Average inference time in one sample: {total_infer_time/len(test_loader.dataset):.8f}s")
    print(f"Average inference time in one batch: {total_infer_time/(len(test_loader.dataset)/args.batch_size):.8f}s")
    print("#"*30)
    print()

    return reachable_point_dict


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    test_set = PotentialFieldDataset(args.data_root, phase=args.phase)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = create_model(args, device)

    criterion = nn.MSELoss()
    # reachable_point_dict = json.load(open("./results/reachable_point_dict.json"))
    with torch.no_grad():
        reachable_point_dict = test(args, device, test_loader, model, criterion)

    with open('./results/reachable_point_dict.json', 'w') as f:
        json.dump(reachable_point_dict, f, indent=4)
    
    new_dict = rearange_dict(reachable_point_dict)
    with open('./results/new_dict.json', 'w') as f:
        json.dump(new_dict, f, indent=4)
