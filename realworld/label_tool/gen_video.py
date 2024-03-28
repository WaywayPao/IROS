import numpy as np
import os
import cv2
import imageio
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from get_data import nuscenes_dataset_helper

title_font = ImageFont.truetype(
    '/home/waywaybao_cs10/Downloads/Roboto/Roboto-Bold.ttf', 70)


def write_text(frame, frame_id, color=(120, 120, 255)):

    title_text = f"Frame: {frame_id:08d}"
    cv2.putText(frame, title_text, org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=1, lineType=1)

    return frame


def make_video(scenarios, video_name, json_name, FPS=2):

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, frameSize=(640, 256))
    box_json = {}

    for i, data in enumerate(scenarios, 1):

        box_json[str(i)] = data["2dbox"]
        src_img = np.array(data["image"])

        for actor_id in data["2dbox"]:
            p1, p2 = data["2dbox"][actor_id]
            src_img = cv2.rectangle(src_img, tuple(p1), tuple(p2), (255, 0, 0), 2) 
            cv2.putText(src_img, text=f"ID:{actor_id}",
                            org=p1,
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1.5,
                            thickness=1,
                            color=(0, 0, 255))


        frame = write_text(src_img, i)
        video.write(frame[:,:,::-1])

    video.release()
    with open (json_name, 'w') as f:
        json.dump(box_json, f, indent=4)


def get_cameras(sample, h=256, w=640, top_crop=0, dataset_dir=None):

    image_path, I_original = sample['images'], sample['intrinsics']

    h_resize = h + top_crop
    w_resize = w

    src_image = Image.open(dataset_dir +'/'+ image_path)
    image = np.array(src_image)
    image_new = src_image.resize((w_resize, h_resize), resample=Image.BILINEAR)
    image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

    return image, image_new, image_path


def get_normalize_box(bbox, h=256, w=640):
    
    box_json = dict()

    for box in bbox:
        cls, [p1, p2], actor_id = box
        p1[0] = int(p1[0]*w/1600+0.5)
        p1[1] = int(p1[1]*h/900+0.5)
        p2[0] = int(p2[0]*w/1600+0.5)
        p2[1] = int(p2[1]*h/900+0.5)
        box_json[actor_id] = [p1, p2]

    return box_json


if __name__ == '__main__':

    phase = 'train'
    dataset_dir = f"/media/waywaybao_cs10/Disk_2/nuscenes_{phase}"
    nusc = nuscenes_dataset_helper(f'v1.0-{phase}val', dataset_dir)
    print("Total scenarios:", len(nusc.nusc.scene))
    
    for i, scene_record in enumerate(nusc.nusc.scene, 1):

        data = []
        file_name = scene_record["name"]
        sample_token = scene_record['first_sample_token']
        
        while sample_token:

            sample_record = nusc.nusc.get('sample', sample_token)
            sample = nusc.parse_sample_record(sample_record,'CAM_FRONT')
            image, image_new, image_path = get_cameras(sample, dataset_dir=dataset_dir)
            box_json = get_normalize_box(sample['bbox'])
            result = {'image': image_new, 'name':image_path, '2dbox': box_json}
            data.append(result)
            sample_token = sample_record['next']
        
        make_video(data, f"./videos/{phase}/"+file_name+'.avi', f"./2dbox/{phase}/"+file_name+'.json')
        print(i, file_name, "Done.")


