import time
import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import parse_args
from models import MODEL
from dataset import PotentialFieldDataset
from test import test


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
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    count_parameters(model)

    return model


def write_result(args, epoch, logs):

    # save training logs in json type
    history_result = json.load(open(args.log_path))
    history_result.append(logs)
    with open(args.log_path, "w") as f:
        json.dump(history_result, f, indent=4)

    print(f"lr: {logs['lr']}")
    print(f"train Loss: {logs['train_loss']:.10f}")
    print("")


def train(args, device, train_loader, validation_loader, model, criterion, optimizer, scheduler):

    logs_list = list()
    start = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        model.train()
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:

            for rgb_inputs, trajs, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch:2d}/{args.epochs:2d}")
                
                rgb_inputs = rgb_inputs.to(device, dtype=torch.float32)
                # traj = traj.to(device, dtype=torch.float32).reshape(-1, 1, 160, 80)
                trajs = trajs.to(device, dtype=torch.float32)

                outputs =  model(rgb_inputs)

                loss = criterion(outputs, trajs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # trajs[:, 0] = trajs[:, 0]*160
                # trajs[:, 1] = trajs[:, 1]*80
                # outputs[:, 0] = outputs[:, 0]*160
                # outputs[:, 1] = outputs[:, 1]*80
                # print("trajs:\t", torch.tensor(trajs, dtype=int))
                # print("outputs:", torch.tensor(outputs, dtype=int))
                # print("#"*20)

                # statistics
                running_loss += loss.item()*outputs.shape[0]
                tepoch.set_postfix(loss=loss.item())
        
        scheduler.step(epoch)
        epoch_loss = running_loss/(len(train_loader.dataset))

        logs = {"epoch":epoch, "lr":scheduler.optimizer.param_groups[0]['lr'], "train_loss":epoch_loss}
        write_result(args, epoch, logs)
        
        with torch.no_grad():
            reachable_point_dict = test(args, device, validation_loader, model, criterion)

        if (epoch+1)%args.save_epoch==0:
            torch.save(model, os.path.join(args.ckpt_folder, f"epoch_{epoch}.pth"))

    elapsed = time.time() - start
    print(f"Training complete in {int(elapsed//60):4d}m {int(elapsed)%60:2d}s")

    return logs_list


if __name__ == '__main__':

    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # prepare dataset
    train_set = PotentialFieldDataset(args.data_root, phase=args.phase)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    validation_set = PotentialFieldDataset(args.data_root, phase='validation')
    validation_loader = DataLoader(dataset=validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = create_model(args, device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-07, amsgrad=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, min_lr=0.000001)

    # train
    train(args, device, train_loader, validation_loader, model, criterion, optimizer, scheduler)
