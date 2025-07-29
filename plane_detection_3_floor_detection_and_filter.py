import numpy as np
from PIL import Image
import os
import scipy
import json
import tqdm

source_root = '/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/mmdetection3d/data/sunrgbd/sunrgbd_trainval'
z_threshold = 0.04  # point cloud z-axis std threshold to select horizontal plane
floor_size_list = {}
surface_size_list = {}  # New: track all horizontal surfaces

# Size thresholds
s_threshold = 0.7739  # small size floor area threshold
m_threshold = 2.5477  # medium size floor area threshold

# New: Height thresholds for detecting different types of surfaces
table_height_range = (0.5, 1.2)  # Tables are typically 0.5-1.2m high
desk_height_range = (0.6, 0.9)   # Desks are typically 0.6-0.9m high
counter_height_range = (0.8, 1.0) # Counters are typically 0.8-1.0m high

for i in range(350):  # for all scenes
    img_id = format(i+1, '06d')
    print('processing {}'.format(img_id))
    
    plane_path = os.path.join(source_root, 'plane/{}/plane_statistics.json'.format(img_id))
    if not os.path.exists(plane_path):
        print('can not find plane results of {}'.format(img_id))
        continue

    # Load plane statistics
    with open(plane_path, 'r') as fh:
        plane_statistics_json = json.load(fh)

    # Find ALL horizontal planes (not just floor)
    horizontal_planes = {}
    floor_index = None
    floor_mean = 10
    
    # First pass: identify all horizontal planes
    for plane_id, plane_sta in plane_statistics_json.items():
        # Skip if plane_sta is not a dictionary (might be metadata)
        if not isinstance(plane_sta, dict):
            continue
            
        # Ensure we have the required fields
        if 'std' not in plane_sta or 'mean' not in plane_sta:
            continue
            
        if plane_sta['std'][2] < z_threshold:  # Small std means horizontal
            horizontal_planes[plane_id] = plane_sta
            
            # Calculate plane properties
            x_min = plane_sta['min'][0]
            x_max = plane_sta['max'][0]
            y_min = plane_sta['min'][1]
            y_max = plane_sta['max'][1]
            z_mean = plane_sta['mean'][2]
            
            area = (x_max - x_min) * (y_max - y_min)
            plane_sta['area'] = area
            
            # Classify plane size
            if area < s_threshold:
                plane_sta['size'] = 's'
            elif area < m_threshold:
                plane_sta['size'] = 'm'
            else:
                plane_sta['size'] = 'l'
            
            # Find the lowest plane as floor
            if z_mean < floor_mean:
                floor_mean = z_mean
                floor_index = plane_id

    if horizontal_planes:
        # Mark the floor
        if floor_index:
            horizontal_planes[floor_index]['type'] = 'floor'
            # Store floor reference separately (not in horizontal_planes)
            floor_info = horizontal_planes[floor_index]
            floor_height = floor_info['mean'][2]
            
            # Second pass: classify other horizontal surfaces based on height
            for plane_id, plane_sta in horizontal_planes.items():
                if plane_id != floor_index and isinstance(plane_sta, dict) and 'type' not in plane_sta:
                    height_above_floor = plane_sta['mean'][2] - floor_height
                    
                    # Classify surface based on height
                    if table_height_range[0] <= height_above_floor <= table_height_range[1]:
                        plane_sta['type'] = 'table'
                        plane_sta['height_above_floor'] = height_above_floor
                    elif desk_height_range[0] <= height_above_floor <= desk_height_range[1]:
                        plane_sta['type'] = 'desk'
                        plane_sta['height_above_floor'] = height_above_floor
                    elif counter_height_range[0] <= height_above_floor <= counter_height_range[1]:
                        plane_sta['type'] = 'counter'
                        plane_sta['height_above_floor'] = height_above_floor
                    elif height_above_floor > 0.2:  # Any horizontal surface above 20cm
                        plane_sta['type'] = 'surface'
                        plane_sta['height_above_floor'] = height_above_floor
                    else:
                        plane_sta['type'] = 'unknown'
            
            # Validate floor (ensure no plane below it)
            VALID_FLOOR = True
            for plane_id, plane_sta in plane_statistics_json.items():
                if isinstance(plane_sta, dict) and 'mean' in plane_sta:
                    if plane_sta['mean'][2] < floor_height - 0.1:
                        VALID_FLOOR = False
                        break
            
            if VALID_FLOOR:
                # Save enhanced plane information
                root_path = os.path.join(source_root, 'plane/{}'.format(img_id))
                
                # Save all horizontal surfaces info
                output_data = {
                    'horizontal_planes': horizontal_planes,
                    'floor_id': floor_index,
                    'surfaces': {}
                }
                
                # Organize surfaces by type
                for plane_id, plane_sta in horizontal_planes.items():
                    if isinstance(plane_sta, dict) and 'type' in plane_sta and plane_sta['type'] != 'floor':
                        surface_type = plane_sta['type']
                        if surface_type not in output_data['surfaces']:
                            output_data['surfaces'][surface_type] = []
                        output_data['surfaces'][surface_type].append({
                            'id': plane_id,
                            'stats': plane_sta
                        })
                
                # Sort surfaces by area (largest first)
                for surface_type in output_data['surfaces']:
                    output_data['surfaces'][surface_type].sort(
                        key=lambda x: x['stats']['area'], 
                        reverse=True
                    )
                
                # Save enhanced plane statistics
                with open(os.path.join(root_path, 'enhanced_plane_statistics.json'), "w") as outfile:
                    json.dump(output_data, outfile, indent=4)
                
                print(f"Scene {img_id}: Found floor and {len(horizontal_planes)-1} other surfaces")
            else:
                print(f'############################### invalid_floor for image {img_id}')
    else:
        print(f'############################### no_horizontal_planes for image {img_id}')

print("Enhanced plane detection complete!")
# DEBUG: Print surface information
print(f"Surface type: {surface_info['type']}")
print(f"Surface center Z: {surface_center_Z}")
print(f"Surface max Z: {surface_info['stats']['max'][2]}")
print(f"Surface min Z: {surface_info['stats']['min'][2]}")
print(f"Surface mean Z: {surface_info['stats']['mean'][2]}")  # Add this line

# Use MEAN instead of MAX for object placement:
inobject.location.x = best_parameter['sample_position_X'] 
inobject.location.y = best_parameter['sample_position_Y']
inobject.location.z = surface_info['stats']['mean'][2] + 0.1  # Changed from max to mean