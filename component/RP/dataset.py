from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import glob
import torch
import numpy as np
import json
import os

class PotentialFieldDataset(Dataset):
    def __init__(self, data_root, phase):
        self.data_root = data_root
        # self.max_step = max_step
        # self.N = max_object_n
        self.sample_list = list()

        if phase == 'train':
            town = ["1_", "2_", "3_", "6_", "7_", "A1"][:]
        elif phase == 'validation':
            town = ["5_"]
        else:
            town = ["10", "A6", "B3"]


        self.final_reachable_points = json.load(open("./utils/final_reachable_points.json"))

        for basic in self.final_reachable_points:
            if not basic[:2] in town:
                continue

            for variant in self.final_reachable_points[basic]:

                for frame_id in self.final_reachable_points[basic][variant]:
                    for actor_id in self.final_reachable_points[basic][variant][frame_id]:
                        self.sample_list.append([basic, variant, frame_id, actor_id])

        self.camera_transforms = transforms.Compose([
            # transforms.Resize(args.img_resize, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[41.2006], std=[19.8283]),
        ])

        self.sample_list = self.sample_list[:]


    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, index):

        basic, variant, frame_id, actor_id = self.sample_list[index]
        reachable_point = self.final_reachable_points[basic][variant][frame_id][actor_id]

        str_frame_id = f"{int(frame_id):08d}"
        npy_path = os.path.join(self.data_root, basic, variant, "actor_pf_npy", str_frame_id+'.npy')
        npy_file = np.load(npy_path, allow_pickle=True).item()

        rgb_img = npy_file[actor_id].clip(0.1, 90)
        rgb_img = np.expand_dims(rgb_img, axis=0)

        # rgb_img = npy_file[actor_id].clip(0.1, 90).T
        # rgb_img = np.expand_dims(rgb_img, axis=2)
        # rgb_img = self.camera_transforms(rgb_img)


        x, y = reachable_point
        # traj= np.array([x/160., y/80.])
        traj= np.array([x, y])

        return rgb_img, traj, [basic, variant, frame_id, actor_id]
