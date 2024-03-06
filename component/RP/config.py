import os
import json
import argparse
from datetime import datetime


def get_current_time():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dtime = datetime.fromtimestamp(timestamp)

    return dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute, dtime.second


def create_ckpt_result(args):

    if args.phase != 'train':
        return args

    copy_args = vars(args).copy()
    log = {"args": copy_args}
    print(log)

    year, month, day, hour, minute, second = get_current_time()
    formated_time = f"{year}-{month}-{day}_{hour:02d}{minute:02d}{second:02d}"
    args.ckpt_folder = os.path.join("./checkpoints", formated_time)
    args.log_path = os.path.join("./logs", f"{formated_time}.json")

    with open(args.log_path, "w") as f:
        json.dump([log], f, indent=4)

    if not os.path.isdir(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)

    return args


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--data_root", type=str, 
                        default="/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data/interactive")
    parser.add_argument('--phase', type=str, choices=['train', 'validation', 'test'], required=True)
    parser.add_argument('--batch_size', default=16,
                        type=int, help='batch size')
    parser.add_argument('--num_workers', default=8,
                        type=int, help='num_workers in dataloader')
    # training
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='initial weight decay')
    parser.add_argument('--max_object_n', default=20, type=int)
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--ckpt_path', default='', type=str)
    parser.add_argument('--save_epoch', default=1, type=int,
                        help='each #epoch save model weights')

    args = parser.parse_args()
    args = create_ckpt_result(args)

    return args