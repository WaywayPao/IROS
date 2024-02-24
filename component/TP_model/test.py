import argparse
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from model import TP_MODEL
from dataset import RiskBenchDataset
from utils import create_folder, count_parameters, write_result


def load_weight(model, checkpoint):
    model = torch.load(checkpoint)
    return copy.deepcopy(model)


def create_data_loader(args):

    dataset_test = RiskBenchDataset(args.data_root, phase='test')

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        # shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Training samples:", len(dataset_test))

    return test_loader


def create_model(args, device):

    model = TP_MODEL().to(device)

    if args.ckpt_path != "":
        model = load_weight(model, args.ckpt_path)
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    count_parameters(model)
    model = model.to(device)

    return model


def test(args, model, test_loader, device):

    criterion = nn.MSELoss()

    start = time.time()
    model.eval()
    dataloader = test_loader

    with torch.no_grad():
        running_loss = 0.0
        with tqdm(dataloader, unit="batch") as tepoch:

            for seg_inputs, gt_tps in tepoch:
                tepoch.set_description(f"Epoch 1/1")
                
                seg_inputs = seg_inputs.to(device, dtype=torch.float32)
                gt_tps = gt_tps.to(device, dtype=torch.float32)

                pred_tps = model(seg_inputs)
                loss = criterion(pred_tps, gt_tps)

                if args.verbose:
                    print(gt_tps[0].tolist(), pred_tps[0].tolist())

                # statistics
                running_loss += loss.item()*pred_tps.shape[0]
                tepoch.set_postfix(loss=loss.item())
        
    elapsed = time.time() - start
    print(f"Training complete in {int(elapsed//60):4d}m {int(elapsed)%60:2d}s")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument('--data_root', type=str, default='/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data')
    parser.add_argument('--ckpts_root', type=str, default='./checkpoints')
    parser.add_argument('--results_root', type=str, default='./results')
    # parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--verbose', action='store_true', default=False)
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # create_folder(args)
    test_loader = create_data_loader(args)
    model = create_model(args, device)

    test(args, model, test_loader, device)
