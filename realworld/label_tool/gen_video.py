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
                fontScale=1, color=color, thickness=2, lineType=1)

    return frame


def make_video(scenarios, video_name, FPS=2):

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, frameSize=(640, 256))

    for i, data in enumerate(scenarios, 1):
        rgb_img = np.array(data["image"])
        frame = write_text(rgb_img, i)
        video.write(frame[:,:,::-1])

    video.release()


def get_cameras(sample, h=256, w=640, top_crop=0):

    image_path, I_original = sample['images'], sample['intrinsics']

    h_resize = h + top_crop
    w_resize = w

    image = Image.open(dataset_dir +'/'+ image_path)
    image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
    image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

    a = cv2.imread(dataset_dir +'/'+ image_path)
    result = {
        'image': image_new,
        'name':image_path,
    }

    return result


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
            result = get_cameras(sample)
            data.append(result)
            sample_token = sample_record['next']
        
        make_video(data, f"./videos/{phase}/"+file_name+'.avi')
        print(i, file_name, "Done.")


