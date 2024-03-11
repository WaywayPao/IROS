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
        self.sample_list = list()

        if phase == 'train':
            town = ["1_", "2_", "3_", "6_", "7_", "A1"][:]
        elif phase == 'validation':
            town = ["5_"]
        else:
            town = ["10", "A6", "B3"]

        self.reachable_points_dict = json.load(open("./utils/new_training_gt_reachable_point.json"))

        for scenario in self.reachable_points_dict:

            if not scenario[:2] in town:
                continue
            
            basic = '_'.join(scenario.split('_')[:-3])
            variant = '_'.join(scenario.split('_')[-3:])
            for frame_id in self.reachable_points_dict[scenario]:
                self.sample_list.append([basic, variant, frame_id])

        # self.camera_transforms = transforms.Compose([
        #     # transforms.Resize(args.img_resize, interpolation=InterpolationMode.NEAREST),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[41.2006], std=[19.8283]),
        # ])

        self.sample_list = self.sample_list[:]


    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, index):

        basic, variant, frame_id = self.sample_list[index]
        reachable_point = self.reachable_points_dict[basic+'_'+variant][frame_id]["all_actor"]

        str_frame_id = f"{int(frame_id):08d}"
        npy_path = os.path.join(self.data_root, basic, "variant_scenario", variant, "actor_pf_npy", str_frame_id+'.npy')
        npy_file = np.load(npy_path, allow_pickle=True).item()
        
        roadline_pf = npy_file['roadline']
        attractive_pf = npy_file['attractive']
        all_actor_pf = npy_file["all_actor"]
        
        gt_pf = (all_actor_pf+roadline_pf+attractive_pf).clip(0.1, 90)
        gt_pf = np.expand_dims(gt_pf, axis=0)

        # gt_pf = npy_file[actor_id].clip(0.1, 90).T
        # gt_pf = np.expand_dims(gt_pf, axis=2)
        # gt_pf = self.camera_transforms(gt_pf)


        x, y = reachable_point
        # traj= np.array([x/160., y/80.])
        traj= np.array([x, y])

        return gt_pf, traj, [basic, variant, frame_id]
