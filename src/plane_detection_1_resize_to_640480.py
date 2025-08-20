import numpy as np
from PIL import Image
import os
import scipy
import json
import tqdm
import argparse

def resize_images(source_root):
    """Resize RGB and depth images to 640x480 resolution."""
    
    # Create output directories if they don't exist
    os.makedirs(os.path.join(source_root, 'image_resize_640480'), exist_ok=True)
    os.makedirs(os.path.join(source_root, 'raw_depth_resize_640480'), exist_ok=True)
    
    rgb_list = os.listdir(os.path.join(source_root, 'image'))
    depth_list = os.listdir(os.path.join(source_root, 'raw_depth'))
    
    for i, rgb_file in enumerate(rgb_list):
        print(i)
        img_id = rgb_file.split('.')[0]
        
        # Forward step
        # RGB
        rgbpath = os.path.join(source_root, 'image/{}'.format(rgb_file))
        rgbpath_new = os.path.join(source_root, 'image_resize_640480/{}'.format(rgb_file))
        
        # Depth
        depthpath = os.path.join(source_root, 'raw_depth/{}.png'.format(img_id))
        depthpath_new = os.path.join(source_root, 'raw_depth_resize_640480/{}.png'.format(img_id))
        
        # Check if files exist before processing
        if os.path.exists(rgbpath) and os.path.exists(depthpath):
            rgb = Image.open(rgbpath)
            depth = Image.open(depthpath)
            
            rgb_new = rgb.resize((640, 480))
            rgb_new.save(rgbpath_new)
            
            depth_new = depth.resize((640, 480))
            depth_new.save(depthpath_new)
        else:
            print(f"Warning: Missing files for {img_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize SUN RGB-D images to 640x480')
    parser.add_argument('--source_root', 
                       default='data/sunrgbd/sunrgbd_trainval',
                       help='Root path of downloaded SUN RGB-D dataset')
    
    args = parser.parse_args()
    resize_images(args.source_root)