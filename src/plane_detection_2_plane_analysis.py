import numpy as np
from PIL import Image
import os
import scipy
import json
import tqdm
import argparse

def process_plane_data(source_root, max_scenes=1000):
    """Process plane detection data for SUN RGB-D scenes."""
    
    for i in range(max_scenes):  # for all 3000, 6000 scenes
        
        img_id = format(i+1, '06d')  # '000063' # '0065', '000063' '010335'
        print('processing {}'.format(img_id))
        
        if os.path.exists(os.path.join(source_root, 'plane/{}/plane_statistics.json'.format(img_id))):
            continue
            
        ''' prepare plane image '''
        # convert plane detection results from 640x480 back to SUN RGB-D size 730x530
        raw_rgbpath = os.path.join(source_root, 'image/{}.jpg'.format(img_id))
        raw_planepath = os.path.join(source_root, 'plane/{}/-plane-opt.png'.format(img_id))
        new_planepath = os.path.join(source_root, 'plane/{}/-plane-opt_final.png'.format(img_id))  # same size as original image
        com_path = os.path.join(source_root, 'plane/{}/-plane-opt_combine.png'.format(img_id))  # to visualize
        
        if not os.path.exists(raw_planepath):
            raw_planepath = raw_planepath.replace('-opt.', '.')
            
        # Check if required files exist
        if not os.path.exists(raw_rgbpath):
            print(f'RGB file not found for {img_id}: {raw_rgbpath}')
            continue
        if not os.path.exists(raw_planepath):
            print(f'Plane file not found for {img_id}: {raw_planepath}')
            continue
            
        raw_rgb = Image.open(raw_rgbpath)
        raw_plane = Image.open(raw_planepath)
        new_plane = raw_plane.resize(raw_rgb.size)
        new_plane.save(new_planepath, format="png")
        
        # paste plane on top of the raw image
        raw_rgb_rgba = raw_rgb.convert("RGBA")
        new_plane_rgba = new_plane.convert("RGBA")
        new_plane_rgba_array = np.array(new_plane_rgba)
        new_plane_rgba_array[:, :, -1] = new_plane_rgba_array[:, :, -1]/3
        new_plane_rgba_tran = Image.fromarray(new_plane_rgba_array)
        raw_rgb.paste(new_plane_rgba_tran, (0, 0), new_plane_rgba_tran)
        raw_rgb.save(com_path, format="png")
        
        ''' point cloud result analysis'''
        ## 1 load plan info'''
        root_path = os.path.join(source_root, 'plane/{}'.format(img_id))
        plane_info_raw_txt = os.path.join(root_path, '-plane-data-opt.txt')
        
        if not os.path.exists(plane_info_raw_txt):
            plane_info_raw_txt = plane_info_raw_txt.replace('-opt.', '.')
            
        if not os.path.exists(plane_info_raw_txt):
            print(f'Plane data file not found for {img_id}: {plane_info_raw_txt}')
            continue
            
        with open(plane_info_raw_txt) as f:
            lines = f.readlines()
            
        plane_info = {}
        for line in lines[1:]:  # first line is legend, last line is background
            info = line.split(' ')
            # plane_index
            plane_index = int(info[0])
            # plane_color_in_png_image
            plane_color_in_png_image = [int(info[2]), int(info[3]), int(info[4])]
            plane_info[plane_index] = plane_color_in_png_image
            
        ## 2 load plane png
        plane_rgb = Image.open(new_planepath)
        plane_rgb_array = np.array(plane_rgb)
        
        ## 3 load saved point cloud'''
        xyz_depth_path = os.path.join(source_root, 'xyz_depth')
        xyz_depth_file = os.path.join(xyz_depth_path, '{}.mat'.format(img_id))
        
        if not os.path.exists(xyz_depth_file):
            print(f'XYZ depth file not found for {img_id}: {xyz_depth_file}')
            continue
            
        xyz_depth = scipy.io.loadmat(xyz_depth_file)
        xyz_depth_array = xyz_depth['instance']
        xyz_depth_array.resize(plane_rgb.size[1], plane_rgb.size[0], 3)  # (x, y, 3)
        
        plane_point_cloud = {}
        plane_mask_disk = {}
        for id in plane_info.keys():
            plane_point_cloud[id] = []
            plane_mask_disk[id] = np.zeros(plane_rgb_array.shape[:2])
            
        for i in range(plane_rgb_array.shape[0]):  # row
            for j in range(plane_rgb_array.shape[1]):  # column
                for plane_id, plane_color in plane_info.items():
                    if list(plane_rgb_array[i, j]) == plane_color:  # match to one plane
                        plane_point_cloud[plane_id].append(xyz_depth_array[i, j])
                        plane_mask_disk[plane_id][i, j] = 1
                        
        ### 4 scene analysis, save mean, std, max, min for selecting horizontal plan later
        plane_statistics = {}
        for id in plane_info.keys():
            plane_statistics[id] = {}
            
        for plane_id, point_cloud in plane_point_cloud.items():  # for each plane
            if len(point_cloud) > 0:  # Check if plane has points
                point_cloud_array = np.array(point_cloud)
                plane_statistics[plane_id]['mean'] = np.mean(point_cloud_array, axis=0).tolist()
                plane_statistics[plane_id]['std'] = np.std(point_cloud_array, axis=0).tolist()
                plane_statistics[plane_id]['max'] = np.max(point_cloud_array, axis=0).tolist()
                plane_statistics[plane_id]['min'] = np.min(point_cloud_array, axis=0).tolist()
            else:
                print(f'Warning: Plane {plane_id} has no points for scene {img_id}')
                
        # save statistics
        with open(os.path.join(root_path, 'plane_statistics.json'), "w") as outfile:
            json.dump(plane_statistics, outfile, indent=4)
            
        # save each plane to double check the correctness
        for id in plane_info.keys():
            plane_array = plane_rgb_array * plane_mask_disk[id][:,:,np.newaxis]
            plane_mask = Image.fromarray(plane_array.astype(np.uint8))
            plane_mask.save(os.path.join(root_path, 'mask_{}.png'.format(id)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SUN RGB-D plane detection data')
    parser.add_argument('--source_root', 
                       default='data/sunrgbd/sunrgbd_trainval',
                       help='Root path of SUN RGB-D dataset')
    parser.add_argument('--max_scenes', type=int, default=1000,
                       help='Maximum number of scenes to process')
    
    args = parser.parse_args()
    process_plane_data(args.source_root, args.max_scenes)