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

    dataset_train = RiskBenchDataset(args.data_root, phase='train')
    dataset_val = RiskBenchDataset(args.data_root, phase='validation')

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        # shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Training samples:", len(dataset_train))
    print("Validation samples:", len(dataset_val))

    return train_loader, validation_loader


def create_model(args, device):

    model = TP_MODEL().to(device)

    if args.ckpt_path != "":
        model = load_weight(model, args.ckpt_path)
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    count_parameters(model)
    model = model.to(device)

    return model


def train(args, model, train_loader, validation_loader, device):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-07, amsgrad=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, min_lr=0.000001)

    logs_list = list()
    start = time.time()

    for epoch in range(args.start_epoch, args.epochs+1):
        epoch_loss = {}

        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = validation_loader
            
            running_loss = 0.0
            with tqdm(dataloader, unit="batch") as tepoch:

                for seg_inputs, gt_tps in tepoch:
                    tepoch.set_description(f"Epoch {epoch:2d}/{args.epochs:2d}")
                    
                    seg_inputs = seg_inputs.to(device, dtype=torch.float32)
                    gt_tps = gt_tps.to(device, dtype=torch.float32)

                    pred_tps = model(seg_inputs)
                    print(gt_tps[0].tolist(), pred_tps[0].tolist())
                    loss = criterion(pred_tps, gt_tps)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item()*pred_tps.shape[0]
                    tepoch.set_postfix(loss=loss.item())
            
            epoch_loss[phase] = running_loss/(len(dataloader.dataset))

        logs = {"epoch":epoch, "lr":scheduler.optimizer.param_groups[0]['lr'], "loss":epoch_loss}
        write_result(args, epoch, logs)
        scheduler.step(epoch)

        torch.save(model, os.path.join(args.ckpts_path, f"epoch_{epoch}.pth"))

    elapsed = time.time() - start
    print(f"Training complete in {int(elapsed//60):4d}m {int(elapsed)%60:2d}s")

    return logs_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument('--data_root', type=str, default='/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data')
    parser.add_argument('--ckpts_root', type=str, default='./checkpoints')
    parser.add_argument('--results_root', type=str, default='./results')
    # parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    create_folder(args)
    train_loader, validation_loader = create_data_loader(args)
    model = create_model(args, device)

    train(args, model, train_loader, validation_loader, device)
