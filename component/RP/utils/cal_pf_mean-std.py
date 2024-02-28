import numpy as np
import json
import os
from collections import OrderedDict


data_root = "../../pf_trajectory_40x20_actor-target=60_v4/interactive"
train_town = ["1_", "2_", "3_", "6_", "7_", "A1", "5_"][:] # 1350, (45, 30)
test_town = ["10", "A6", "B3"]   # 515, (47, 11)
town =  train_town[:-1]

def main():

    cnt = 0
    n_sample = 0
    pf_list = list()

    for basic in sorted(os.listdir(data_root)):
        if not basic[:2] in town:
            continue
        basic_path = os.path.join(data_root, basic)

        for variant in sorted(os.listdir(basic_path)):
            variant_path = os.path.join(basic_path, variant, 'actor_pf_npy')

            for frame in sorted(os.listdir(variant_path)):

                npy_path = os.path.join(variant_path, frame)
                npy_file = np.load(npy_path, allow_pickle=True).item()

                for i, actor_id in enumerate(npy_file.keys()):

                    pf_npy = npy_file[actor_id].reshape(-1).clip(0.1, 90)
                    pf_list.append(pf_npy)
                    n_sample += 1

            cnt += 1
            print(cnt, basic, variant, 'Done!!!')

    pf_npy = np.stack(pf_list)
    print("pf_npy shape:", pf_npy.shape)

    channel_mean = pf_npy.mean(1).sum(0) / n_sample
    channel_std = pf_npy.std(1).sum(0) / n_sample

    print("mean:", channel_mean, "\tstd:", channel_std)


if __name__ == '__main__':
    main()


