import os
import numpy as np
from datetime import datetime
import json


def get_current_time():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dtime = datetime.fromtimestamp(timestamp)

    return dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute, dtime.second


def create_folder(args):

    year, month, day, hour, minute, second = get_current_time()
    formated_time = f"{year}-{month}-{day}_{hour:02d}{minute:02d}{second:02d}"

    args.ckpts = f'{args.ckpts}/{formated_time}'
    # args.log_dir = f'{args.log_dir}/{formated_time}'
    args.results = f'{args.results}/{formated_time}.json'

    os.makedirs(f'{args.ckpts}')
    # os.makedirs(f'{args.log_dir}')

    logs = {"args": vars(args).copy()}
    with open(args.results, "w") as f:
        json.dump([logs], f, indent=4)


def count_parameters(model):

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total parameters: ', params)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)


def write_result(args, epoch, logs):

    # save training logs in json type
    history_result = json.load(open(args.results))
    history_result.append(logs)

    with open(args.results, "w") as f:
        json.dump(history_result, f, indent=4)

    print(f"lr: {logs['lr']}")
    print(f"train Loss: {logs['loss']['train_loss']:.10f}")
    print(f"validation Loss: {logs['loss']['validation']:.10f}")
    print("")
