from nuscenes.nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion
import torch
from PIL import Image
import torchvision
from pathlib import Path


def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t
    return pose


def get_pose(rotation, translation, inv=False, flat=False):
    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix
    t = np.array(translation, dtype=np.float32)
    return get_transformation_matrix(R, t, inv=inv)


class nuscenes_dataset_helper:
    
    def __init__(self, version, root):
        self.dataset_dir = Path(root)
        self.nusc = NuScenes(version=version, dataroot=root)
        self.transform = torchvision.transforms.ToTensor()
    
    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)
    
    def parse_sample_record(self, sample_record, camera_rig='CAM_FRONT'):   
        
        sensor_record = self.nusc.get('sample_data', sample_record['data'][camera_rig])
        calibrated_sensor = self.nusc.get('calibrated_sensor', sensor_record['calibrated_sensor_token'])
        ego_sensor = self.nusc.get('ego_pose', sensor_record['ego_pose_token'])
        world_from_ego = self.parse_pose(ego_sensor)
        ego_from_world = self.parse_pose(ego_sensor, inv=True)
        sensor_from_ego = self.parse_pose(calibrated_sensor, inv=True)
        intrinsic = calibrated_sensor['camera_intrinsic']
        extrinsic = sensor_from_ego @ ego_from_world

        sample = {'images':sensor_record['filename'],
                  'intrinsics': intrinsic,
                   'extrinsics': extrinsic}

        return sample


    def get_cameras(self, sample, h=256, w=640, top_crop=0):
        """
        Note: we invert I and E here for convenience.
        """
        images = list()
        intrinsics = list()

        image_path, I_original = sample['images'], sample['intrinsics']

        # for image_path, I_original in zip(sample['images'], sample['intrinsics']):
        h_resize = h + top_crop
        w_resize = w
        # tensor = torch.load(self.dataset_dir / tensor_path)
        image = Image.open(self.dataset_dir / image_path)
        image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
        image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))
        I = np.float32(I_original)
        I[0, 0] *= w_resize / image.width
        I[0, 2] *= w_resize / image.width
        I[1, 1] *= h_resize / image.height
        I[1, 2] *= h_resize / image.height
        I[1, 2] -= top_crop
        
        images.append(self.transform(image_new))
        intrinsics.append(torch.tensor(I))


        result = {
            'image': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(sample['extrinsics'])),
            'name':image_path,
        }


        return result

    def iter_nusc(self):
        # Iterate dataset's scene
        # Each scene contains multiple sample(frame)
        all_data = []
        
        for i, scene_record in enumerate(self.nusc.scene):
            data = []
            sample_token = scene_record['first_sample_token']
            while sample_token:
                sample_record = self.nusc.get('sample', sample_token)
                sample = self.parse_sample_record(sample_record,'CAM_FRONT')
                result = nusc.get_cameras(sample)
                data.append(result)
                sample_token = sample_record['next']

            all_data.append(data)
            # break

        return all_data


if __name__ == '__main__':

    # nusc = nuscenes_dataset_helper('v1.0-test', "/media/waywaybao_cs10/Disk_2/nuscenes_test")
    nusc = nuscenes_dataset_helper('v1.0-trainval', "/media/waywaybao_cs10/Disk_2/nuscenes_train")
    all_data = nusc.iter_nusc()
    print(len(all_data))