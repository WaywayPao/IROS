import numpy as np
import torch
import json
import os
import cv2
from riskbench_data import BirdViewProducer, BirdView
from mask import PixelDimensions, Loc

N_CLASSES = 7
IMG_H = 200
IMG_W = 200
PIXELS_PER_METER = 4
ZOOM_IN_PEDESTRIAN = 4

root_dir = "/media/waywaybao_cs10/DATASET/RiskBench_Dataset"
data_type = ["interactive", "non-interactive", "obstacle", "collision"][:]


def preprocess_semantic(topdown):
    
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown


def get_item(variant_path, frame, actor_attribute_data, data_type="interactive"):
    

    ego_data_path = os.path.join(variant_path, "ego_data")    
    ego_data = json.load(open(os.path.join(ego_data_path, frame+'.json')))
    ego_pos = Loc(x=ego_data["location"]["x"], y=ego_data["location"]["y"])
    ego_yaw = ego_data["rotation"]["yaw"]

    actor_data_path = os.path.join(variant_path, "actors_data")
    actor_data = json.load(open(os.path.join(actor_data_path, frame+'.json')))

    # Draw BEV map 
    obstacle_bbox_list = []
    pedestrian_bbox_list = []
    vehicle_bbox_list = []
    agent_bbox_list = []
            

    ego_id = actor_attribute_data["ego_id"]
    # if scneario is non-interactive, obstacle --> interactor_id is -1 
    interactor_id = actor_attribute_data["interactor_id"] 
    vehicle_id_list = list(actor_attribute_data["vehicle"].keys())
    pedestrian_id_list = list(actor_attribute_data["pedestrian"].keys())
    obstacle_id_list = list(actor_attribute_data["obstacle"].keys())

    # obstacle bbox store in actor_attribute.json
    for actor_id in obstacle_id_list:
        pos_0 = actor_attribute_data["obstacle"][str(actor_id)]["cord_bounding_box"]["cord_0"]
        pos_1 = actor_attribute_data["obstacle"][str(actor_id)]["cord_bounding_box"]["cord_4"]
        pos_2 = actor_attribute_data["obstacle"][str(actor_id)]["cord_bounding_box"]["cord_6"]
        pos_3 = actor_attribute_data["obstacle"][str(actor_id)]["cord_bounding_box"]["cord_2"]
        
        obstacle_bbox_list.append([actor_id, 
                                   [Loc(x=pos_0[0], y=pos_0[1]), 
                                    Loc(x=pos_1[0], y=pos_1[1]), 
                                    Loc(x=pos_2[0], y=pos_2[1]), 
                                    Loc(x=pos_3[0], y=pos_3[1]),]
                                    ])


    for actor_id in vehicle_id_list:
        if not "cord_bounding_box" in actor_data[str(actor_id)]:
            continue

        pos_0 = actor_data[str(actor_id)]["cord_bounding_box"]["cord_0"]
        pos_1 = actor_data[str(actor_id)]["cord_bounding_box"]["cord_4"]
        pos_2 = actor_data[str(actor_id)]["cord_bounding_box"]["cord_6"]
        pos_3 = actor_data[str(actor_id)]["cord_bounding_box"]["cord_2"]
        
        # print(actor_id)

        if int(actor_id) == int(ego_id):
            # print("ego actor_id ")                
            agent_bbox_list.append([actor_id, 
                                    [Loc(x=pos_0[0], y=pos_0[1]),
                                     Loc(x=pos_1[0], y=pos_1[1]), 
                                     Loc(x=pos_2[0], y=pos_2[1]), 
                                     Loc(x=pos_3[0], y=pos_3[1]),]
                                     ])
        
        elif int(actor_id) == int(interactor_id):
            
            if data_type != "collision":
            
                vehicle_bbox_list.append([actor_id,
                                          [Loc(x=pos_0[0], y=pos_0[1]), 
                                           Loc(x=pos_1[0], y=pos_1[1]), 
                                           Loc(x=pos_2[0], y=pos_2[1]), 
                                           Loc(x=pos_3[0], y=pos_3[1]),]
                                           ])

        else:
            vehicle_bbox_list.append([actor_id, 
                                      [Loc(x=pos_0[0], y=pos_0[1]), 
                                       Loc(x=pos_1[0], y=pos_1[1]),
                                       Loc(x=pos_2[0], y=pos_2[1]),
                                       Loc(x=pos_3[0], y=pos_3[1]),]
                                       ])


    for actor_id in pedestrian_id_list:
        if not "cord_bounding_box" in actor_data[str(actor_id)]:
            continue
        pos_0 = actor_data[str(actor_id)]["cord_bounding_box"]["cord_0"]
        pos_1 = actor_data[str(actor_id)]["cord_bounding_box"]["cord_4"]
        pos_2 = actor_data[str(actor_id)]["cord_bounding_box"]["cord_6"]
        pos_3 = actor_data[str(actor_id)]["cord_bounding_box"]["cord_2"]        

        if ZOOM_IN_PEDESTRIAN != 1:
            loc = actor_data[str(actor_id)]["location"]
            x, y = loc["x"], loc["y"]
            
            for pos in [pos_0, pos_1, pos_2, pos_3]:
                vec = (pos[0]-x, pos[1]-y)
                pos[0] = x+ZOOM_IN_PEDESTRIAN*vec[0]
                pos[1] = y+ZOOM_IN_PEDESTRIAN*vec[1]


        if int(actor_id) == int(interactor_id):
            
            if data_type != "collision":
                pedestrian_bbox_list.append([actor_id,
                                             [Loc(x=pos_0[0], y=pos_0[1]), 
                                              Loc(x=pos_1[0], y=pos_1[1]), 
                                              Loc(x=pos_2[0], y=pos_2[1]), 
                                              Loc(x=pos_3[0], y=pos_3[1]),]
                                              ])
                
        else:
            
            pedestrian_bbox_list.append([actor_id, 
                                         [Loc(x=pos_0[0], y=pos_0[1]), 
                                          Loc(x=pos_1[0], y=pos_1[1]), 
                                          Loc(x=pos_2[0], y=pos_2[1]), 
                                          Loc(x=pos_3[0], y=pos_3[1]),]
                                        ])


    return ego_pos, ego_yaw, obstacle_bbox_list, pedestrian_bbox_list, vehicle_bbox_list, agent_bbox_list


def gen_BEV(variant_path, frame, Town, data_type="interactive"):
    
    
    actor_attribute_path = os.path.join(variant_path, "actor_attribute.json")
    actor_attribute_data = json.load(open(actor_attribute_path))

    ego_pos, ego_yaw, obstacle_bbox_list, pedestrian_bbox_list, vehicle_bbox_list, agent_bbox_list = \
        get_item(variant_path, frame, actor_attribute_data, data_type=data_type)
    

    birdview_producer = BirdViewProducer(
                            Town, 
                            PixelDimensions(width=IMG_W, height=IMG_H), 
                            pixels_per_meter=PIXELS_PER_METER)
    

    birdview: BirdView = birdview_producer.produce(ego_pos, yaw=ego_yaw,
                                                    agent_bbox_list=agent_bbox_list, 
                                                    vehicle_bbox_list=vehicle_bbox_list,
                                                    pedestrians_bbox_list=pedestrian_bbox_list,
                                                    obstacle_bbox_list=obstacle_bbox_list)


    birdview_rgb = BirdViewProducer.as_rgb(birdview)
    birdview_mask =  BirdViewProducer.as_ss(birdview)
    # topdown = preprocess_semantic(birdview_mask).numpy()

    # print(np.max(birdview_mask))
    # print(birdview_mask.shape)
    

    # visualize
    # for c in range(7):
    #     # a = np.array((256,256), dtype=np.uint8)
    #     a = np.where(birdview_mask==c, 1, 0)
    #     cv2.imwrite(f"birdview_mask_{c}.png", a*255)

    # a = cv2.imread(variant_path+'/rgb/front/'+frame+'.jpg')
    # b = cv2.imread(variant_path+'/instance_segmentation/top/'+frame+'.png')
    cv2.imwrite("birdview.png", birdview_rgb[:, :, ::-1])
    # cv2.imwrite("src_front.png", a)
    # cv2.imwrite("src_top.png", b)

    return birdview_mask.astype(np.uint8)
    # return topdown.astype(np.bool_)


def main(scenario_list, cpu_id):
    
    for _type, basic, variant, town in scenario_list:

        variant_dir = os.path.join(data_root, basic, "variant_scenario", variant)
        N = len(os.listdir(variant_dir+"/ego_data"))
        
        save_dir = os.path.join(target_root, basic, "variant_scenario", variant, 'bev-seg')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # if len(os.listdir(save_dir)) == N:
        #     continue
        
        for frame_id in range(1, N+1):
            if frame_id == 37:
                bev_seg = gen_BEV(variant_dir, f"{frame_id:08d}", town, _type)
            # np.save(save_dir+f"/{frame_id:08d}.npy", bev_seg)

        print(
            f"CPU ID: {cpu_id:3d}\t{_type}\t{basic}_{variant}\t Done!")


if __name__ == '__main__':

    for _type in data_type:

        data_root = os.path.join(root_dir, _type)
        target_root = os.path.join("/media/waywaybao_cs10/DATASET/RiskBench_Dataset/other_data", _type)
        scenario_list = []

        for basic in sorted(os.listdir(data_root)):
            town = basic[:2]
            if town == '10':
                town = 'Town10HD'                
            elif town in ["1_", "2_", "3_", "5_", "6_", "7_"]:
                town = f'Town0{town[0]}'

            basic_path = os.path.join(data_root, basic, "variant_scenario")

            for variant in sorted(os.listdir(basic_path)):
                if not (basic == "10_t2-2_0_c_l_r_1_0" and variant == "CloudySunset_low_"):
                    continue
                scenario_list.append([_type, basic, variant, town])

        main(scenario_list, 0)

        # from multiprocessing import Pool
        # from multiprocessing import cpu_count

        # cpu_n = 15
        # variant_per_cpu = len(scenario_list)//cpu_n+1
        # pool_sz = cpu_count()

        # with Pool(pool_sz) as p:
        #     res = p.starmap(main, [(scenario_list[i*variant_per_cpu:i*variant_per_cpu+variant_per_cpu], i)
        #                     for i in range(cpu_n)])
        #     p.close()
        #     p.join()

        