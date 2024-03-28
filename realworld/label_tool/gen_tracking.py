import cv2
import json
from get_data import nuscenes_dataset_helper
from gen_video import get_cameras, get_normalize_box, make_video
from ultralytics import YOLO

"""
    https://docs.ultralytics.com/zh/modes/track/#available-trackers
"""
# Load the YOLOv8 model
model = YOLO('./yolov8x.pt')
# model = YOLO('./yolov8n.pt')

def get_box(boxes_info):

    corner_list = []

    for box_info in boxes_info:
        cls = int(box_info["class"])
        actor_id = int(box_info["track_id"])
        box = box_info["box"]

        if not cls in [0,1,2,3,5,7]:
            continue
        else:
            cls = 'human' if cls==0 else 'vehicle'

        p1 = [box["x1"], box["y1"]]
        p2 = [box["x2"], box["y2"]]

        corner_list.append([cls, [p1, p2], actor_id])
        
    return corner_list


def run_track(frame, is_new_scenario=False, frame_cnt=None):

    results = model.track(frame, persist=is_new_scenario, verbose=False, augment=True, classes=[0,1,2,3,5,7])
    json_result = json.loads(results[0].tojson())

    if results[0].boxes.is_track:
        corner_list = get_box(json_result)
    else:
        corner_list = []

    return corner_list


if __name__ == '__main__':

    phase = 'test'
    dataset_dir = f"/media/waywaybao_cs10/Disk_2/nuscenes_{phase}"
    nusc = nuscenes_dataset_helper(f'v1.0-{phase}', dataset_dir)
    print("Total scenarios:", len(nusc.nusc.scene))
    
    for i, scene_record in enumerate(nusc.nusc.scene, 1):

        if i > 4 :
            continue
        
        data = []
        file_name = scene_record["name"]
        sample_token = scene_record['first_sample_token']
        is_first_frame = True
        frame_cnt = 1

        while sample_token:

            sample_record = nusc.nusc.get('sample', sample_token)
            sample = nusc.parse_sample_record(sample_record,'CAM_FRONT')
            image, image_new, image_path = get_cameras(sample, dataset_dir=dataset_dir)
            bbox = run_track(image, is_new_scenario=is_first_frame, frame_cnt=frame_cnt)
            box_json = get_normalize_box(bbox)

            result = {'image': image_new, 'name':image_path, '2dbox': box_json}
            data.append(result)
            
            sample_token = sample_record['next']
            is_first_frame = False
            frame_cnt += 1

        make_video(data, f"./videos/{phase}/"+file_name+'.avi', f"./2dbox/{phase}/"+file_name+'.json')
        print(i, file_name, "Done.")

