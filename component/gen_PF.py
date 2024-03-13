import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
import cv2
import torch
import PIL.Image as Image
from collections import OrderedDict
from plantcv import plantcv as pcv
# Set global debug behavior to None (default), "print" (to file), or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = None


IMG_H = 100
IMG_W = 200

PIX_PER_METER = 4
sx = IMG_W//2
sy = 3*PIX_PER_METER    # the distance from the ego's center to his head

# create mask for data preprocessing
VIEW_MASK_CPU = cv2.imread("../utils/mask_120degree.png")
# VIEW_MASK_CPU = np.ones((100, 200, 3), dtype=np.uint8)*255
VIEW_MASK_CPU = (VIEW_MASK_CPU[:100,:,0] != 0).astype(np.float32)
VIEW_MASK = torch.from_numpy(VIEW_MASK_CPU).cuda(0)
VIEW_MASK_IDX = torch.where(VIEW_MASK != 0)
VIEW_MASK_IDX = torch.cat((VIEW_MASK_IDX[0].unsqueeze(-1), VIEW_MASK_IDX[1].unsqueeze(-1)), 1)
VIEW_MASK_Y, VIEW_MASK_X = VIEW_MASK_IDX.T.long()
NEW_VIEW_MASK = torch.zeros((IMG_H, IMG_W, 2)).cuda(0)+0.01
for (y,x) in VIEW_MASK_IDX:
    NEW_VIEW_MASK[y, x] = torch.tensor([y,x])


def get_seg_mask(raw_bev_seg, use_gt=False, channel=5):

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
            PEDESTRIANS = 3
            VEHICLES = 2
            ROAD_LINE = 1
            ROAD = 0
    """

    if use_gt:
        new_bev_seg = np.where((raw_bev_seg<6) & (raw_bev_seg>0), raw_bev_seg, 0)
        new_bev_seg = torch.LongTensor(new_bev_seg)
        one_hot = torch.nn.functional.one_hot(new_bev_seg, channel+1).float()
        
        return one_hot[:,:,1:].numpy()*VIEW_MASK_CPU[:,:,None]
    
    else:
        one_hot = raw_bev_seg*VIEW_MASK_CPU[:,:,None]
        return one_hot


def create_roadline_pf(bev_seg, ROBOT_RAD=2.0, KR=400.0):

    road = bev_seg[:, :, 0]
    roadline = bev_seg[:, :, 1]
    vehicle = bev_seg[:, :, 2]
    pedestrian = bev_seg[:, :, 3]
    road = ((road+vehicle+pedestrian)!=0).cpu().numpy()

    canny = cv2.Canny(road.astype(np.uint8), 0, 1)
    roadline = torch.from_numpy(roadline + canny)

    oy, ox = torch.where(roadline != 0)
    if len(oy) == 0:
        oy, ox = [0], [0]
    obstacle_tensor = torch.from_numpy(np.stack((oy, ox), 1)).cuda(0)
    
    roadline_pf = create_repulsive_potential(obstacle_tensor, ROBOT_RAD=ROBOT_RAD, KR=KR)

    return roadline_pf


def create_attractive_pf(goal_tensor, KP=0.75, EPS=0.01):

    # attractive_pf = KP * np.hypot(x - gx, y - gy)
    attractive_pf = (KP*torch.sum((NEW_VIEW_MASK - goal_tensor)**2, axis=2)**0.5) * VIEW_MASK

    return attractive_pf


def create_repulsive_potential(obstacle_tensor, ROBOT_RAD=5.0, KR=1000.0):
    
    repulsive_pf = torch.zeros((IMG_H, IMG_W), dtype=torch.float32).cuda(0)

    # find the minimum distance min_dq to obstacle for each pixel
    dis_list = torch.sum((VIEW_MASK_IDX[:, None, :] - obstacle_tensor)**2, axis=2)
    min_dq = (torch.min(dis_list, axis=1)[0])**0.5

    # if min_dq is smaller than the given radius, a repulsive force is assigned, otherwise 0
    # repulsive force = KR / ((min_dq + 2) ** 2)
    mask = min_dq <= ROBOT_RAD*10
    repulsive_pf[VIEW_MASK_Y[mask], VIEW_MASK_X[mask]] = KR / ((min_dq[mask] + 2) ** 2)

    return repulsive_pf


def Render_PF(gx, gy, bev_seg_list):

    gx = sx+int(gx+0.5)
    gy = IMG_H-int(gy+0.5)
    goal_tensor = torch.tensor([gy, gx]).cuda(0)
    pf_list = list()

    for bev_seg in bev_seg_list:

        roadline_pf = create_roadline_pf(bev_seg.copy(), ROBOT_RAD=2.0, KR=400.0)
        attractive_pf = create_attractive_pf(goal_tensor)

        pf = OrderedDict()
        pf['all_actor'] = torch.zeros((IMG_H, IMG_W), dtype=torch.float32)
        pf['roadline'] = roadline_pf
        pf['attractive'] = attractive_pf

        obstacle_mask = ((bev_seg[:,:,2]+bev_seg[:,:,3])*255).astype(torch.uint8)

        if torch.sum(obstacle_mask) < 20:
            clust_masks = []
        else:
            _, clust_masks = pcv.spatial_clustering(mask=obstacle_mask, algorithm="DBSCAN", min_cluster_size=5, max_distance=0.5)

        oy_list = [0]
        ox_list = [0]
        idx = 0

        for mask in clust_masks:

            oy, ox = np.where(mask!=0)
            if len(oy) < 100:
                continue

            oy_list.extend(oy)
            ox_list.extend(ox)

            obstacle_tensor = torch.from_numpy(np.stack((oy, ox), 1)).cuda(0)

            actor_pf = create_repulsive_potential(obstacle_tensor, ROBOT_RAD=5.0, KR=1000.0)
            pf[idx] = actor_pf
            idx += 1

        obstacle_tensor = torch.from_numpy(np.stack((oy_list, ox_list), 1)).cuda(0)    
        all_actor_pf = create_repulsive_potential(obstacle_tensor, ROBOT_RAD=5.0, KR=1000.0)
        pf['all_actor'] = all_actor_pf

        pf_list.append(pf)

    return pf_list, goal_tensor

