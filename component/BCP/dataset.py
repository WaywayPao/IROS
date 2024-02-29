import os
import torch
import torch.utils.data as data
import numpy as np
import PIL.Image as Image
import cv2
import random
import json
import time

__all__ = [
    'VisionDataLayer',
    'BEV_SEGDataLayer',
    'PFDataLayer',
]

# TBD
class VisionDataLayer(data.Dataset):

    def __init__(self, data_root, behavior_root, scenario, time_steps, camera_transforms, num_box,
                 raw_img_size=(256, 640), img_resize=(256, 640), data_augmentation=True, phase="train"):

        self.data_root = data_root
        self.time_steps = time_steps
        self.camera_transforms = camera_transforms
        self.raw_img_size = raw_img_size
        self.img_resize = img_resize
        self.data_augmentation = data_augmentation
        self.phase = phase
        self.num_box = num_box
        self.scale_w = img_resize[1]/raw_img_size[1]
        self.scale_h = img_resize[0]/raw_img_size[0]

        self.behavior_dict = {}
        self.state_dict = {}
        self.cnt_labels = np.zeros(2, dtype=np.int32)
        self.inputs = []

        start_time = time.time()

        self.load_behavior(behavior_root)
        # self.load_state(state_root, scenario)
        self.target_points = json.load(open("./utils/goal_list.json"))


        for (basic, variant, data_type) in scenario:

            variant_path = os.path.join(
                data_root, data_type, basic, "variant_scenario", variant)

            # get positive and negative behavior from behavior_dict
            frames, labels = self.get_behavior(
                basic, variant, variant_path, data_type)

            for frame_no, label in list(zip(frames, labels))[::5]:
                # for frame_no, label in zip(frames_no, labels):
                self.inputs.append([variant_path, frame_no-self.time_steps+1,
                                    frame_no+1, np.array(label, dtype=np.bool), data_type])
                self.cnt_labels[int(label)] += 1

        end_time = time.time()

        """
            train   Label 'go'   (negative):   38387
            train   Label 'stop' (positive):   27032
            test    Label 'go'   (negative):   11072
            test    Label 'stop' (positive):    4148
        """
        print(f"{phase}\tLabel 'go'   (negative): {self.cnt_labels[0]:7d}")
        print(f"{phase}\tLabel 'stop' (positive): {self.cnt_labels[1]:7d}")
        print(f"Load datas in {end_time-start_time:4.4f}s")
        print()

        self.inputs = self.inputs[:]


    def load_behavior(self, behavior_root):

        for _type in ["interactive", "obstacle"]:
            behavior_path = os.path.join(
                behavior_root, f"{_type}.json")
            behavior = json.load(open(behavior_path))
            self.behavior_dict[_type] = behavior

    def load_state(self, state_root, scenario):
        
        for (basic, variant, data_type) in scenario:
            state_path = os.path.join(state_root, data_type, basic+"_"+variant+".json")
            state = json.load(open(state_path))   
            self.state_dict[basic+"_"+variant] = state

    def get_behavior(self, basic, variant, variant_path, data_type, start_frame=1):

        N = len(os.listdir(variant_path+'/rgb/front'))
        first_frame_id = start_frame + self.time_steps - 1
        last_frame_id = start_frame + N - 1

        frames = list(range(N+1))
        labels = np.zeros(N+1)

        # frames = list(range(first_frame_id, last_frame_id+1))
        # labels = np.zeros(N, dtype=np.int32)

        if data_type in ["interactive", "obstacle"]:
            stop_behavior = self.behavior_dict[data_type][basic][variant]
            start, end = stop_behavior
            start = max(start, first_frame_id)
            end = min(end, last_frame_id)

            labels[start: end+1] = 1.

        frames = frames[first_frame_id:]
        labels = labels[first_frame_id:]

        return frames, labels

    def normalize_box(self, trackers):
        """
            return normalized_trackers TxNx4 ndarray:
            [BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_BOTRIGHT_X, BBOX_BOTRIGHT_Y]
        """

        normalized_trackers = trackers.copy()
        normalized_trackers[:, :,
                            0] = normalized_trackers[:, :, 0] * self.scale_w
        normalized_trackers[:, :,
                            2] = normalized_trackers[:, :, 2] * self.scale_w
        normalized_trackers[:, :,
                            1] = normalized_trackers[:, :, 1] * self.scale_h
        normalized_trackers[:, :,
                            3] = normalized_trackers[:, :, 3] * self.scale_h

        return normalized_trackers

    def process_tracking(self, variant_path, start, end):
        """
            tracking_results Kx10 ndarray:
            [FRAME_ID, ACTOR_ID, BBOX_TOPLEFT_X, BBOX_TOPLEFT_Y, BBOX_WIDTH, BBOX_HEIGHT, 1, -1, -1, -1]
            e.g. tracking_results = np.array([[187, 876, 1021, 402, 259, 317, 1, -1, -1, -1]])
        """

        INTENTIONS = {'r': 1, 'sl': 2, 'f': 3, 'gi': 4, 'l': 5, 'gr': 6, 'u': 7, 'sr': 8,'er': 9}

        def parse_scenario_id(variant_path):
            
            variant_path_token = variant_path.split('/')
            
            basic = variant_path_token[-3]
            variant = variant_path_token[-1]

            basic_token = basic.split('_')

            if "obstacle" in variant_path:
                return basic_token[3], basic, variant
            else:
                return basic_token[5], basic, variant

        ego_intention, basic, variant = parse_scenario_id(variant_path)

        tracking_results = np.load(
            os.path.join(variant_path, 'tracking.npy'))
        assert len(tracking_results) > 0, f"{variant_path} No tracklet"

        height, width = self.raw_img_size

        t_array = tracking_results[:, 0]
        tracking_index = tracking_results[np.where(t_array == end-1)[0], 1]

        trackers = np.zeros([self.time_steps, self.num_box, 4]).astype(np.float32) # TxNx4
        intentions = np.zeros(10).astype(np.float32)   # 10
        # states = np.zeros([self.time_steps, self.num_box+1, 2]).astype(np.float32)   # Tx(N+1)x2

        gx, gy = self.target_points[basic+'_'+variant][f"{end-1:08d}"]
        target_point = np.array([(gx-80)/80, (80-gy)/80], dtype=np.float32)

        intentions[INTENTIONS[ego_intention]] = 1

        for t in range(start, end):
            current_tracking = tracking_results[np.where(t_array == t)[0]]

            for i, object_id in enumerate(tracking_index):
                current_actor_id_idx = np.where(
                    current_tracking[:, 1] == object_id)[0]

                if len(current_actor_id_idx) != 0:
                    # x1, y1, x2, y2
                    bbox = current_tracking[current_actor_id_idx, 2:6]
                    bbox[:, 0] = np.clip(bbox[:, 0], 0, width)
                    bbox[:, 2] = np.clip(bbox[:, 0]+bbox[:, 2], 0, width)
                    bbox[:, 1] = np.clip(bbox[:, 1], 0, height)
                    bbox[:, 3] = np.clip(bbox[:, 1]+bbox[:, 3], 0, height)
                    trackers[t-start, i, :] = bbox

                    # try:
                    #     states[t-start, i+1] = self.state_dict[basic+"_"+variant][str(t)][str(object_id)]
                    # except:
                    #     states[t-start, i+1] = 0


        trackers = self.normalize_box(trackers)

        return trackers, tracking_index, intentions, target_point


    def __getitem__(self, index):

        variant_path, start, end, label, data_type = self.inputs[index]

        camera_inputs = []

        for idx in range(start, end):
            camera_name = f"{int(idx):08d}.jpg"
            camera_path = os.path.join(variant_path, "rgb/front", camera_name)
            img = self.camera_transforms(
                Image.open(camera_path).convert('RGB').copy())
            camera_inputs.append(img)

        camera_inputs = torch.stack(camera_inputs)

        trackers, tracking_id, intention_inputs, target_point = self.process_tracking(
            variant_path, start, end)

        # add data augmentation
        mask = torch.ones(
            (self.time_steps, 3, self.img_resize[0], self.img_resize[1]))

        return camera_inputs, trackers, mask, label, intention_inputs, target_point, 0, 0, 0, end-1

    def __len__(self):
        return len(self.inputs)


class BEV_SEGDataLayer(data.Dataset):
    
    def __init__(self, img_root, behavior_root, num_box=25,\
                  img_resize=(100, 200), camera_transforms=None, use_gt=False, time_step=5, phase="train"):

        if phase == 'train':
            town = ["1_", "2_", "3_", "6_", "7_", "A1"][:]
        elif phase == 'validation':
            town = ["5_"]
        else:
            town = ["10", "A6", "B3"]

        self.img_root = img_root
        self.num_box = num_box
        self.img_resize = img_resize
        self.use_gt = use_gt
        self.time_step = time_step
        self.data_types = ["interactive", "non-interactive", "obstacle", "collision"][:1]

        self.VIEW_MASK = (cv2.imread("../../utils/mask_120degree.png")[:,:,0] != 0).astype(np.float32)
        self.target_points = {}
        self.behavior_dict = {}
        self.load_behavior(behavior_root)

        skip_list = json.load(open("../../utils/skip_scenario.json"))
        self.cnt_labels = np.zeros(2, dtype=np.int32)

        self.inputs = []
        start_time = time.time()

        for data_type in self.data_types:
            if self.use_gt:
                self.target_points[data_type] = json.load(open(f"../../utils/target_point_{data_type}.json"))
            else:
                self.target_points[data_type] = None
        
            type_path = os.path.join(img_root, data_type)

            for basic in sorted(os.listdir(type_path)):
                if not basic[:2] in town:
                    continue
                basic_path = os.path.join(type_path, basic, 'variant_scenario')

                for variant in os.listdir(basic_path):
                    if [data_type, basic, variant] in skip_list:
                        continue
                    
                    seg_folder = os.path.join(basic_path, variant, "bev-seg")
                    N = len(os.listdir(seg_folder))
                    frames, labels = self.get_behavior(data_type, basic, variant, N)

                    for frame_no, label in list(zip(frames, labels))[self.time_step:-20:self.time_step]:
                        self.inputs.append([data_type, basic, variant, frame_no, np.array(label, dtype=np.float32)])
                        self.cnt_labels[int(label)] += 1

        end_time = time.time()

        print(f"{phase}\tLabel 'go'   (negative): {self.cnt_labels[0]:7d}")
        print(f"{phase}\tLabel 'stop' (positive): {self.cnt_labels[1]:7d}")
        print(f"Load datas in {end_time-start_time:4.4f}s")
        print()

        self.inputs = self.inputs[:]


    def load_behavior(self, behavior_root):

        for _type in ["interactive", "obstacle"]:
            behavior_path = os.path.join(
                behavior_root, f"{_type}.json")
            behavior = json.load(open(behavior_path))
            self.behavior_dict[_type] = behavior


    def get_behavior(self, data_type, basic, variant, N, start_frame=1):

        first_frame_id = start_frame + self.time_step - 1
        last_frame_id = start_frame + N - 1

        frames = list(range(N+1))
        labels = np.zeros(N+1)

        if data_type in ["interactive", "obstacle"]:
            stop_behavior = self.behavior_dict[data_type][basic][variant]
            start, end = stop_behavior
            start = max(start, first_frame_id)
            end = min(end, last_frame_id)

            labels[start: end+1] = 1.

        frames = frames[first_frame_id:]
        labels = labels[first_frame_id:]

        return frames, labels


    def onehot_seg(self, bev_seg, N_CLASSES=5):
        
        TARGET = {"roadway":[43,255,123], "roadline":[255,255,255], "vehicle":[120, 2, 255], "pedestrian":[222,134,120]}
        """
            src:
                AGENT = 6
                OBSTACLES = 5
                PEDESTRIANS = 4
                VEHICLES = 3
                ROAD_LINE = 2
                ROAD = 1
                UNLABELES = 0
            new (return):
                OBSTACLES = 4
                PEDESTRIANS = 3
                VEHICLES = 2
                ROAD_LINE = 1
                ROAD = 0
        """

        if self.use_gt:
            new_bev_seg = np.where((bev_seg<6) & (bev_seg>0), bev_seg, 0)
            new_bev_seg = torch.LongTensor(new_bev_seg)
            one_hot = torch.nn.functional.one_hot(new_bev_seg, N_CLASSES+1).permute(2, 0, 1).float()

            return one_hot[1:]
        
        else:
            new_bev_seg = np.zeros((self.img_resize[0], self.img_resize[1], N_CLASSES+1), dtype=np.float32)

            for idx, cls in enumerate(TARGET):
                target_color = np.array(TARGET[cls])
                matching_pixels = np.all(bev_seg == target_color, axis=-1).astype(np.float32)
                new_bev_seg[:, :, idx] = matching_pixels*self.VIEW_MASK[:100]

            one_hot = torch.LongTensor(new_bev_seg).permute(2, 0, 1).float()
            return one_hot[:-1]


    def __getitem__(self, index):

        data_type, basic, variant, frame, label = self.inputs[index]
        variant_path = os.path.join(self.img_root, data_type, basic, "variant_scenario", variant)
        
        gt_seg_list = []

        for frame_id in range(frame-self.time_step+1, frame+1):

            if self.use_gt:
                seg_path = os.path.join(variant_path, "bev-seg", f"{frame_id:08d}.npy")
                gt_seg = (np.load(seg_path)*self.VIEW_MASK)[:100]

            else:
                seg_path = os.path.join(variant_path, "cvt_bev-seg", f"{frame_id:08d}.png")
                gt_seg = (np.array(Image.open(seg_path).convert('RGB').copy()))[:100]

            # new_gt_seg : Cx100x200
            new_gt_seg = self.onehot_seg(gt_seg)
            gt_seg_list.append(new_gt_seg)

        gt_seg_list = torch.stack(gt_seg_list)

        target_point = self.target_points[data_type][basic+'_'+variant][f"{frame:08d}"]
        target_point = torch.Tensor(target_point)

        # padding tracker (useless)
        trackers = np.zeros([self.time_step, self.num_box, 4]).astype(np.float32)
        mask = torch.ones((self.time_step, 4, self.img_resize[0], self.img_resize[1]))

        return gt_seg_list, target_point, trackers, mask, label

    def __len__(self):
        return len(self.inputs)


class PFDataLayer(data.Dataset):
    
    def __init__(self, img_root, behavior_root, num_box=25,\
                  img_resize=(100, 200), camera_transforms=None, use_gt=False, time_step=5, phase="train"):

        if phase == 'train':
            town = ["1_", "2_", "3_", "6_", "7_", "A1"][:]
        elif phase == 'validation':
            town = ["5_"]
        else:
            town = ["10", "A6", "B3"]

        self.img_root = img_root
        self.num_box = num_box
        self.img_resize = img_resize
        self.use_gt = use_gt
        self.time_step = time_step
        self.data_types = ["interactive", "non-interactive", "obstacle", "collision"][:1]

        # self.VIEW_MASK = (cv2.imread("../../utils/mask_120degree.png")[:,:,0] != 0).astype(np.float32)
        self.target_points = {}
        self.behavior_dict = {}
        self.load_behavior(behavior_root)

        skip_list = json.load(open("../../utils/skip_scenario.json"))
        self.cnt_labels = np.zeros(2, dtype=np.int32)

        self.inputs = []
        start_time = time.time()

        for data_type in self.data_types:
            self.target_points[data_type] = json.load(open(f"../../utils/target_point_{data_type}.json"))
            type_path = os.path.join(img_root, data_type)

            for basic in sorted(os.listdir(type_path)):
                if not basic[:2] in town:
                    continue
                basic_path = os.path.join(type_path, basic, 'variant_scenario')

                for variant in os.listdir(basic_path):
                    if [data_type, basic, variant] in skip_list:
                        continue
                    
                    seg_folder = os.path.join(basic_path, variant, "bev-seg")
                    N = len(os.listdir(seg_folder))
                    frames, labels = self.get_behavior(data_type, basic, variant, N)

                    if phase == 'test':
                        frame_list = list(zip(frames, labels))[self.time_step-1::]
                    else:
                        frame_list = list(zip(frames, labels))[self.time_step-1:-20:self.time_step]
                    
                    for frame_no, label in frame_list:
                        self.inputs.append([data_type, basic, variant, frame_no, np.array(label, dtype=np.float32)])
                        self.cnt_labels[int(label)] += 1

        end_time = time.time()

        print(f"{phase}\tLabel 'go'   (negative): {self.cnt_labels[0]:7d}")
        print(f"{phase}\tLabel 'stop' (positive): {self.cnt_labels[1]:7d}")
        print(f"Load datas in {end_time-start_time:4.4f}s")
        print()

        self.inputs = self.inputs[:]


    def load_behavior(self, behavior_root):

        for _type in ["interactive", "obstacle"]:
            behavior_path = os.path.join(
                behavior_root, f"{_type}.json")
            behavior = json.load(open(behavior_path))
            self.behavior_dict[_type] = behavior


    def get_behavior(self, data_type, basic, variant, N, start_frame=1):

        first_frame_id = start_frame + self.time_step - 1
        last_frame_id = start_frame + N - 1

        frames = list(range(N+1))
        labels = np.zeros(N+1)

        if data_type in ["interactive", "obstacle"]:
            stop_behavior = self.behavior_dict[data_type][basic][variant]
            start, end = stop_behavior
            start = max(start, first_frame_id)
            end = min(end, last_frame_id)

            labels[start: end+1] = 1.

        frames = frames[first_frame_id:]
        labels = labels[first_frame_id:]

        return frames, labels


    def __getitem__(self, index):

        data_type, basic, variant, frame, label = self.inputs[index]
        variant_path = os.path.join(self.img_root, data_type, basic, "variant_scenario", variant)
        gt_pf_list = []

        for frame_id in range(frame-self.time_step+1, frame+1):

            if self.use_gt:
                pf_path = os.path.join(variant_path, "actor_pf_npy", f"{frame_id:08d}.npy")
            else:
                pf_path = os.path.join(variant_path, "pre_cvt_actor_pf_npy", f"{frame_id:08d}.npy")

            npy_file = np.load(pf_path, allow_pickle=True).item()
            actor_pf = npy_file['all_actor']
            roadline_pf = npy_file['roadline']
            attractive_pf = npy_file['attractive']

            gt_pf = (actor_pf+roadline_pf+attractive_pf).clip(0.1, 90)
            gt_pf = np.expand_dims(gt_pf, 0)
            gt_pf_list.append(torch.from_numpy(gt_pf))

        gt_pf_list = torch.stack(gt_pf_list)

        target_point = self.target_points[data_type][basic+'_'+variant][f"{frame:08d}"]
        target_point = torch.Tensor(target_point)

        # padding tracker (useless)
        trackers = np.zeros([self.time_step, self.num_box, 4]).astype(np.float32)
        mask = torch.ones((self.time_step, 4, self.img_resize[0], self.img_resize[1]))

        return gt_pf_list, target_point, trackers, mask, label

    def __len__(self):
        return len(self.inputs)
