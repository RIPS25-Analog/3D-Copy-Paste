import os
import numpy as np
import scipy
import random
import json


root_path = '/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/mmdetection3d/data/sunrgbd'
scene_pointcloud_files = 'sunrgbd_trainval/pointcloud'
output_path = '/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/3D-Copy-Paste/SUNRGBD_objects_statistic.json'

with open('/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/objaverse/objaverse.json') as f:
    objaverse_objects_dict = json.load(f)
interested = list(objaverse_objects_dict.keys())
print('Interested objects:', interested)

interested_objects = {
    k: []
    for k in interested
}

# training data information
with open(os.path.join(root_path, 'sunrgbd_trainval/train_data_idx.txt')) as f:
    train_data_ids_raw = f.readlines()

train_data_ids = ["{:06d}".format(int(id.split('\n')[0])) for id in train_data_ids_raw]  # change format


for scene_id in train_data_ids: # do the first 10 for debug, for each scene
    print('Finish image id ', scene_id)


    with open(os.path.join(root_path, 'sunrgbd_trainval/label', '{}.txt'.format(scene_id))) as f:
        lines = f.readlines()

    ori_GT_object_info_dict = {}
    ori_GT_object_info_list = []

    if len(lines) == 0:  # do not exist objects, load a templete
        with open(os.path.join(root_path, 'sunrgbd_trainval/label', '000063.txt'.format(scene_id))) as f:
            lines = f.readlines()

    for line in lines:  # for each GT object
        raw_info_list = [line.split('\n')[0].split(' ')][0]
        class_name = raw_info_list[0]
        info_list = [float(ele) for ele in raw_info_list[1:]]
        if class_name not in ori_GT_object_info_dict.keys():  # each object insert only one
            ori_GT_object_info_dict[class_name] = info_list
            ori_GT_object_info_list.append(info_list)  # easy to compute min max
    for obj_class, obj_info in ori_GT_object_info_dict.items():
        if obj_class in interested:
            interested_objects[obj_class].append(obj_info[9]) # only about height, the size

# save
interested_objects_statistics = {}
for key, value in interested_objects.items():
    print(key, value)
    if len(value) == 0:
        continue
    interested_objects_statistics[key] = []
    interested_objects_statistics[key].append(np.mean(value))
    interested_objects_statistics[key].append(np.std(value))

with open(output_path, 'w') as outfile:
    json.dump(interested_objects_statistics, outfile, indent=4)
