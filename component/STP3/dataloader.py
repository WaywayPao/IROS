import os
import torch
import json
import numpy as np
from collections import OrderedDict


"""
    from dataloader import RiskBenchDataset
    train_dataset = RiskBenchDataset(phase="train")
    validation_dataset = RiskBenchDataset(phase="validation")
    test_dataset = RiskBenchDataset(phase="test")
"""

class RiskBenchDataset(torch.utils.data.Dataset):
    
    def __init__(self, phase=None, future_n=10, ego_data_root="./RiskBench_Dataset", 
                 seg_root="./RiskBench_Dataset/other_data",
                 target_point_root="./RiskBench_Dataset"):
        super(RiskBenchDataset, self).__init__()

        assert phase != None

        if phase == 'train':
            town = ["1_", "2_", "3_", "6_", "7_", "A1"][:]
        elif phase == 'validation':
            town = ["5_"]
        else:
            town = ["10", "A6", "B3"]

        self.future_n = future_n
        self.ego_data_root = ego_data_root
        self.seg_root = seg_root
        self.data_types = ["interactive", "non-interactive", "obstacle", "collision"][:1]
        self.target_points = {}
        
        self.sample_list = []
        for data_type in self.data_types:
            self.target_points[data_type] = json.load(open(f"{target_point_root}/target_point_{data_type}.json"))
        
            type_path = os.path.join(self.ego_data_root, data_type)

            for basic in sorted(os.listdir(type_path)):
                if not basic[:2] in town:
                    continue
                basic_path = os.path.join(type_path, basic, 'variant_scenario')

                for variant in os.listdir(basic_path):
                    img_folder = os.path.join(basic_path, variant, 'rgb/front')
                    for frame in sorted(os.listdir(img_folder))[:-self.future_n]:
                        self.sample_list.append([data_type, basic, variant, int(frame.split('.')[0])])

        self.sample_list = self.sample_list[:]


    def onehot(self, bev_seg, N_CLASSES=6):

        """
            # AGENT = 6 (EGO)
            OBSTACLES = 5
            PEDESTRIANS = 4
            VEHICLES = 3
            ROAD_LINE = 2
            ROAD = 1
            UNLABELES = 0
        """

        bev_seg = np.where((bev_seg<6) & (bev_seg>0), bev_seg, 0)
        bev_seg = torch.LongTensor(bev_seg)
        one_hot = torch.nn.functional.one_hot(bev_seg, N_CLASSES).permute(2, 0, 1).float()

        return one_hot[1:]


    def related_coordinate(self, related_vec, R, PIX_PER_METER=4, sy=12):
        '''
            Cartesian coordinate system, related to ego (0, 0)
        '''
        related_dis_rotation = related_vec@R

        _gx, _gy = related_dis_rotation
        gx = _gx*PIX_PER_METER
        gy = _gy*PIX_PER_METER+sy

        return int(gx+0.5), int(gy+0.5)
    

    def get_future_traj(self, ego_data_path, frame_id):

        future_traj_list = []
        egos_data = OrderedDict()

        for frame in range(frame_id, frame_id+self.future_n):

            jsom_name = f"{frame:08d}.json"
            json_path = os.path.join(ego_data_path, jsom_name)
            egos_data[frame] = json.load(open(json_path))

        ego_data = egos_data[frame_id]
        ego_loc = np.array(
            [ego_data["location"]["x"], ego_data["location"]["y"]])
        theta = ego_data["compass"]
        theta = np.array(theta*np.pi/180.0)
        R = np.array([[np.cos(theta), np.sin(theta)],
                        [np.sin(theta), -np.cos(theta)]])

        for future_frame_id in range(frame_id, frame_id+self.future_n):
            
            future_ego_data = egos_data[future_frame_id]
            future_loc = [future_ego_data["location"]["x"], future_ego_data["location"]["y"]]
            related_vec = np.array(future_loc)-ego_loc            
            new_x, new_y = self.related_coordinate(related_vec, R)

            future_traj_list.append([new_x, new_y, 0])
  
        future_traj_list = torch.Tensor(future_traj_list)
        return future_traj_list


    def __getitem__(self, index):
        '''
            input:
                trajs: torch.Tensor<float> (B, N, n_future, 2)   N: sample number
                semantic_pred: torch.Tensor<float> (B, n_future, 200, 200)
                drivable_area: torch.Tensor<float> (B, 1/2, 200, 200)
                lane_divider: torch.Tensor<float> (B, 1/2, 200, 200)
                target_points: torch.Tensor<float> (B, 2)

            output:
                final_trajectory (B, n_future, 3)
        ''' 
        data_type, basic, variant, frame_id = self.sample_list[index]

        # get trajs
        # TODO
        trajs = ...

        # get semantic_pred, drivable_area, lane_divider
        bev_seg_path = os.path.join(self.seg_root, data_type, basic, "variant_scenario", variant, 'bev-seg', f"{frame_id:08d}.npy")
        bev_seg = (np.load(bev_seg_path))[:100]
        # onehot_bev_seg: 5x100x200
        onehot_bev_seg = self.onehot(bev_seg)

        semantic_pred = onehot_bev_seg[2:4]
        drivable_area = onehot_bev_seg[:1]
        lane_divider = onehot_bev_seg[1:2]

        # get target_point
        x, y = self.target_points[data_type][basic+'_'+variant][f"{frame_id:08d}"]
        x = min(max(-90, x), 90)
        y = min(max(10, y), 90)
        target_point = torch.Tensor([x, y])

        # get final_trajectory
        ego_data_path = os.path.join(self.ego_data_root, data_type, basic, "variant_scenario", variant, "ego_data")
        final_trajectory = self.get_future_traj(ego_data_path, frame_id)

        return trajs, semantic_pred, drivable_area, lane_divider, target_point, final_trajectory


    def __len__(self):
        return len(self.sample_list)



