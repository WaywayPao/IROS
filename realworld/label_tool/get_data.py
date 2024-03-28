from nuscenes.nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion
import torch
from PIL import Image
import torchvision
from pathlib import Path
from nuscenes.utils.data_classes import Box
from shapely.geometry import Point, MultiPoint, box

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
        self.instance_dict = self.get_instance_dict()

    def get_instance_dict(self):
        instance_dict = {}
        i = 1
        for instance in self.nusc.instance:
            instance_token = instance['token']
            if instance_token not in instance_dict.keys():
                instance_dict[instance_token] = i
                i = i + 1
        return instance_dict

    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)
    
    def parse_sample_record(self, sample_record, camera_rig='CAM_FRONT'):   

        sensor_record = self.nusc.get('sample_data', sample_record['data'][camera_rig])
        calibrated_sensor = self.nusc.get('calibrated_sensor', sensor_record['calibrated_sensor_token'])
        ego_sensor = self.nusc.get('ego_pose', sensor_record['ego_pose_token'])
        # world_from_ego = self.parse_pose(ego_sensor)
        ego_from_world = self.parse_pose(ego_sensor, inv=True)
        sensor_from_ego = self.parse_pose(calibrated_sensor, inv=True)
        intrinsic = calibrated_sensor['camera_intrinsic']
        extrinsic = sensor_from_ego @ ego_from_world

        bbox = self.get_box_coor(sample_record, sensor_record)
        sample = {'images':sensor_record['filename'],
                  'intrinsics': intrinsic,
                   'extrinsics': extrinsic,
                   'bbox':bbox}

        return sample

    def get_cameras(self, sample, h=256, w=640, top_crop=0):
        """
        Note: we invert I and E here for convenience.
        """
        images = list()
        intrinsics = list()

        image_path, I_original = sample['images'], sample['intrinsics']

        h_resize = h + top_crop
        w_resize = w

        bbox = sample['bbox']
        box_json = dict()

        for box in bbox:
            cls, [p1, p2], actor_id = box
            box_json[actor_id] = [p1, p2]

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
            '2dbox': box_json,
            'name':image_path,
        }


        return result

    def view_points(self, points, view) -> np.ndarray:
        """
            This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
            orthographic projections. It first applies the dot product between the points and the view. By convention,
            the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
            normalization along the third dimension.
            For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
            For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
            For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
            all zeros) and normalize=False
            :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
            :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
                The projection should be such that the corners are projected onto the first 2 axis.
            :param normalize: Whether to normalize the remaining coordinate (along the third axis).
            :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
        """
        view = np.array(view)

        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[0] == 3

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view
        nbr_points = points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
        
        return points
    
    def get_box_coor(self, sample, cam):

        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        corner_list = []

        for annotation_token in sample['anns']:
            
            # GLOBAL box
            ann = self.nusc.get('sample_annotation', annotation_token)
            obj_cls = ann["category_name"]
            if not ("vehicle" in obj_cls or "human" in obj_cls):
                continue
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            
            # First step: transform from global into the ego vehicle frame for the timestamp of the image.
            box.translate(-np.array(poserecord['translation']))
            box.rotate(Quaternion(poserecord['rotation']).inverse)
            
            # Second step: transform from ego into the camera.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)
            
            # Third step: Project to image -> 8 corners points
            corners_3d = box.corners()
            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            corners_3d = corners_3d[:, in_front]
            corner_coords = self.view_points(corners_3d, np.array(cs_record['camera_intrinsic'])).T[:, :2].tolist()
            final_coords = self.post_process_coords(corner_coords)

            visibility_token = ann['visibility_token']
            visibility = self.nusc.get('visibility', visibility_token)['token']

            if visibility=='4' and final_coords!=None:
                actor_id = self.instance_dict[ann['instance_token']]
                corner_list.append([obj_cls, final_coords, actor_id])    

        return corner_list


    def post_process_coords(self, corner_coords, imsize = (1600, 900)):
        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, imsize[0], imsize[1])

        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            if max_x-min_x>20 and max_y-min_y>20:
                return [[int(min_x+0.5), int(min_y+0.5)], [int(max_x+0.5), int(max_y+0.5)]]
            
        return None


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
                result = self.get_cameras(sample)
                data.append(result)
                sample_token = sample_record['next']

            all_data.append(data)

        return all_data

if __name__ == '__main__':

    # Nusc = nuscenes_dataset_helper('v1.0-test', "/media/waywaybao_cs10/Disk_2/nuscenes_test")
    Nusc = nuscenes_dataset_helper('v1.0-trainval', "/media/waywaybao_cs10/Disk_2/nuscenes_train")
    all_data = Nusc.iter_nusc()
    print("Total #sample:", len(all_data))