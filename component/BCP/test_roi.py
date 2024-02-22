
from datasets import VisionDataLayer, BEV_SEGDataLayer, PFDataLayer
from models import GCN as Model
import utils as utl
import config as cfg

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch, gc
import numpy as np
import json
import copy
import time
import os


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

    state_dict = torch.load(checkpoint)
    state_dict_copy = {}
    for key in state_dict.keys():
        state_dict_copy[key[7:]] = state_dict[key]

    model.load_state_dict(state_dict_copy)
    return copy.deepcopy(model)


def create_model(args, device):

    model = Model(args.method, args.time_steps, pretrained=args.pretrained,
                  partialConv=args.partial_conv, use_intention=args.use_intention, NUM_BOX=args.num_box)

    if args.ckpt_path != "":
        model = load_weight(model, args.ckpt_path)

    model = nn.DataParallel(model).to(device)
    count_parameters(model)

    return model


def create_data_loader(args):

    if args.method == "vision":
        camera_transforms = transforms.Compose([
            # transforms.Resize(args.img_resize, interpolation=InterpolationMode.NEAREST),
            transforms.PILToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        camera_transforms = transforms.Compose([
            # transforms.Resize(args.img_resize, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    DataLayer = {"vision":VisionDataLayer, "bev_seg":BEV_SEGDataLayer, "pf":PFDataLayer}[args.method]

    data_sets = {
        phase: DataLayer(
            data_root=args.data_root,
            behavior_root=args.behavior_root,
            state_root=args.state_root,
            scenario=getattr(args, phase+"_session_set"),
            camera_transforms=camera_transforms,
            num_box=args.num_box,
            # raw_img_size=args.img_size,
            # img_resize=args.img_resize,
            time_steps=args.time_steps,
            data_augmentation=args.data_augmentation,
            phase=phase,
        )
        for phase in args.phases
    }

    data_loaders = {
        phase: DataLoader(
            data_sets[phase],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for phase in args.phases
    }

    return data_loaders


def write_result(args, epoch, pred_metrics, target_metrics, losses, loss_go, loss_stop):

    result = {}
    result['Epoch'] = epoch
    result['lr'] = args.lr
    result['loss_weights'] = args.loss_weights

    for phase in args.phases:
        phase_result = utl.compute_result(
            args.class_index, pred_metrics[phase], target_metrics[phase])

        phase_result['loss'] = losses[phase].item() / \
            len(data_loaders[phase].dataset)
        phase_result['loss_go'] = loss_go[phase].item()  / \
            sum(phase_result['confusion_matrix'][0])
        phase_result['loss_stop'] = loss_stop[phase].item()  / \
            sum(phase_result['confusion_matrix'][1])

        result[phase] = phase_result

        print("#"*40)
        print(f"Phase : {phase}")
        for key in phase_result:
            print(f"{key:16s} : {phase_result[key]}")
        print("#"*40)
        print()

    return result


def BCELoss(pred, gt, weights, reduction='none'):

    go = torch.eq(gt, 0).float()
    stop = torch.eq(gt, 1).float()

    weight = weights[0]*go + weights[1]*stop

    return nn.BCELoss(weight=weight, reduction=reduction)(pred, gt)


def test(args, model, data_loaders, device):

    weights = args.loss_weights
    criterion = BCELoss

    epoch = 1
    phase = 'test'
    
    losses = {phase: 0.0}
    loss_go = {phase: 0.0}
    loss_stop = {phase: 0.0}
    pred_metrics = {phase: []}
    target_metrics = {phase: []}
    ROI_result_dict = OrderedDict()
    ROI_result_dict_score = OrderedDict()

    start = time.time()

    with tqdm(data_loaders[phase], unit="batch") as tepoch:
        for batch_idx, (camera_inputs, trackers, mask, vel_targets, intention_inputs, target_point, \
                        basic_batch, variant_batch, actor_id_batch, frame_no_batch) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch:2d}/{args.epochs:2d}")

            batch_size = camera_inputs.shape[0]
            # BxTxCxHxW
            camera_inputs = to_device(camera_inputs, device)
            # BxTxCxHxW
            mask = to_device(mask, device)
            # BxTxNx4
            trackers = to_device(trackers, device)
            # Bx1
            gt_targets = to_device(vel_targets.float(), device).reshape(-1)
            # Bx10
            intention_inputs = to_device(intention_inputs, device)
            # BxTxNx2
            target_point = to_device(target_point, device)

            pred = model(camera_inputs, trackers, mask,
                        intention_inputs, target_point, device)

            # dynamic loss weight
            loss = criterion(pred, gt_targets, weights, reduction='none')

            loss_go[phase] += torch.sum(loss[torch.where(gt_targets==0)])
            loss_stop[phase] += torch.sum(loss[torch.where(gt_targets==1)])
            losses[phase] += torch.sum(loss)

            loss = torch.mean(loss)

            if args.verbose:
                # print(pred)
                pred_vel = torch.where(pred > 0.5, 1., 0.)
                print(pred_vel)
                print(gt_targets)
                print()

            pred = pred.detach().to('cpu').numpy().reshape((-1, 1))
            pred = np.concatenate((1-pred, pred), 1)
            vel_targets = vel_targets.detach().to('cpu').numpy()
            
            pred_metrics[phase].extend(pred)
            target_metrics[phase].extend(vel_targets)

            for i in range(batch_size):
                basic = basic_batch[i]
                variant = variant_batch[i]
                actor_id = actor_id_batch[i]
                if str(actor_id) == "0":
                    continue
                frame_no = frame_no_batch[i].item()

                if not basic+"_"+variant in ROI_result_dict:
                    ROI_result_dict[basic+"_"+variant] = OrderedDict()
                    ROI_result_dict_score[basic+"_"+variant] = OrderedDict()
                if not str(frame_no) in ROI_result_dict[basic+"_"+variant]:
                    ROI_result_dict[basic+"_"+variant][str(frame_no)] = OrderedDict()
                    ROI_result_dict_score[basic+"_"+variant][str(frame_no)] = OrderedDict()

                ROI_result_dict[basic+"_"+variant][str(frame_no)][str(actor_id)] = bool(pred[i, 1].item() >= 0.5)
                ROI_result_dict_score[basic+"_"+variant][str(frame_no)][str(actor_id)] = pred[i, 1].item()

            tepoch.set_postfix(loss=loss.item())

    end = time.time()

    result = write_result(args, epoch, pred_metrics,
                            target_metrics, losses, loss_go, loss_stop)

    print(f"Epoch {epoch:2d} | validation loss: {result['test']['loss']:.5f} mAP: {result['test']['mAP']:.5f} \
          | running time: {end-start:.2f} sec")

    return ROI_result_dict, ROI_result_dict_score


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', choices=["vision", "bev_seg", "pf"], type=str, required=True)
    parser.add_argument('--data_type', default='all', type=str, required=True)
    parser.add_argument('--phases', default=['test'], type=list)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-07, type=float)
    parser.add_argument('--weight_decay', default=1e-02, type=float)
    parser.add_argument('--loss_weights', default=[1.0, 1.7], type=list)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--time_steps', default=5, type=int)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--partial_conv', default=True, type=bool)
    parser.add_argument('--use_intention', action='store_true', default=False)
    parser.add_argument('--data_augmentation', default=True, type=bool)
    parser.add_argument('--ckpt_path', default="", type=str)
    parser.add_argument('--Method', default="", type=str)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--save_roi', action='store_true', default=False)

    args = cfg.parse_args(parser)
    args = cfg.read_data(args, valiation=False)


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = create_model(args, device)
    model.train(False)

    data_loaders = create_data_loader(args)

    with torch.no_grad():
        ROI_result_dict, ROI_result_dict_score = test(args, model, data_loaders, device)

    if args.save_roi:
        cfg.create_ROI_result(args)
        with open(args.roi_path, "w") as f:
            json.dump(ROI_result_dict, f, indent=4)
        with open(args.roi_root+"/raw_score-"+args.roi_path.split('/')[-1], "w") as f:
            json.dump(ROI_result_dict_score, f, indent=4)

