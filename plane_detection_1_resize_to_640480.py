import numpy as np
from PIL import Image
import os
import scipy
import json
import tqdm

# root path of downloaed SUN RGB-D
source_root = '/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/mmdetection3d/data/sunrgbd/sunrgbd_trainval'

rgb_list = os.listdir(os.path.join(source_root, 'image'))
depth_list = os.listdir(os.path.join(source_root, 'raw_depth'))
for i, rgb_file in enumerate(rgb_list):
    print(i)
    img_id = rgb_file.split('.')[0]
    ''' Forward step'''
    # rgb
    rgbpath = os.path.join(source_root, 'image/{}'.format(rgb_file))
    rgbpath_new = os.path.join(source_root, 'image_resize_640480/{}'.format(rgb_file))
    # depth
    depthpath = os.path.join(source_root, 'raw_depth/{}.png'.format(img_id))
    depthpath_new = os.path.join(source_root, 'raw_depth_resize_640480/{}.png'.format(img_id))

    rgb = Image.open(rgbpath)
    depth = Image.open(depthpath)

    rgb_new = rgb.resize((640, 480))
    rgb_new.save(rgbpath_new)
    depth_new = depth.resize((640, 480))
    depth_new.save(depthpath_new)