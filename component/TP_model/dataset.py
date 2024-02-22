import json
import os
import numpy as np
import torch
import cv2
from PIL import Image


class RiskBenchDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, split='train', time_step=5):
        super(RiskBenchDataset, self).__init__()

        if split == 'train':
            town = ["1_", "2_", "3_", "5_", "6_", "7_", "A1"][:]
        elif split == 'validation':
            town = ["5_"]
        else:
            town = ["10", "A6", "B3"]

        self.img_root = img_root
        self.time_step = time_step
        self.data_types = ["interactive", "non-interactive", "obstacle", "collision"][:3]
        
        self.target_points = json.load(open("../../utils/goal_list.json"))
        skip_list = json.load(open("../../utils/skip_scenario.json"))
        self.VIEW_MASK = cv2.imread("../../utils/VIEW_MASK.png")[:,:,0].copy()
        self.img_list = []
        
        for data_type in self.data_types:
            type_path = os.path.join(img_root, data_type)

            for basic in sorted(os.listdir(type_path)):
                if not basic[:2] in town:
                    continue
                basic_path = os.path.join(type_path, basic, 'variant_scenario')

                for variant in os.listdir(basic_path):
                    if [data_type, basic, variant] in skip_list:
                        continue
                    seg_folder = os.path.join(basic_path, variant, "bev-seg")                    

                    for frame in sorted(os.listdir(seg_folder))[self.time_step::self.time_step]:
                        self.img_list.append([data_type, basic, variant, frame])

        self.img_list = self.img_list[:]


    def seg_onehot(self, bev_seg, N_CLASSES=5):

        """
            AGENT = 6
            OBSTACLES = 5
            PEDESTRIANS = 4
            VEHICLES = 3
            ROAD_LINE = 2
            ROAD = 1
            UNLABELES = 0
        """

        bev_seg = np.where(bev_seg<6, bev_seg, 0)
        topdown = torch.LongTensor(topdown)
        topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

        new_bev_seg = torch.from_numpy(...)
        return new_bev_seg


    def __getitem__(self, index):
        
        data_type, basic, variant, frame = self.img_list[index]
        gt_seg_list = []
        
        start_frame = int(frame.split('.')[0])
        for frame_id in range(start_frame-self.time_step+1, start_frame+1):

            seg_path = os.path.join(self.img_root, data_type, basic, "variant_scenario", variant, 'bev-seg', f"{frame_id:08d}.npy")
            gt_seg = (np.load(seg_path)*self.VIEW_MASK)[:100]
            gt_seg = self.seg_onehot(gt_seg)
            
            gt_seg_list.append(gt_seg)

        gt_seg_list = torch.stack(gt_seg_list)

        target_point = self.target_points[basic+'_'+variant][frame.split('.')[0]]
        target_point = torch.Tensor(target_point)

        return gt_seg_list, target_point

    def __len__(self):
        return len(self.img_list)
