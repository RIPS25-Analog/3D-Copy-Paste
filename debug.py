import json
import os

# Path to check plane statistics structure
source_root = '/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/mmdetection3d/data/sunrgbd/sunrgbd_trainval'

# Check first few scenes
for i in range(1, 5):
    img_id = format(i, '06d')
    plane_path = os.path.join(source_root, 'plane/{}/plane_statistics.json'.format(img_id))
    
    if os.path.exists(plane_path):
        print(f"\n{'='*50}")
        print(f"Checking structure for scene {img_id}")
        print(f"{'='*50}")
        
        with open(plane_path, 'r') as fh:
            plane_statistics_json = json.load(fh)
        
        print(f"Type of loaded JSON: {type(plane_statistics_json)}")
        print(f"Keys in JSON: {list(plane_statistics_json.keys())[:5]}...")  # First 5 keys
        
        # Check the structure of the first item
        for key, value in list(plane_statistics_json.items())[:2]:
            print(f"\nKey: {key}")
            print(f"Type of value: {type(value)}")
            if isinstance(value, dict):
                print(f"Value keys: {list(value.keys())}")
                if 'mean' in value:
                    print(f"Mean type: {type(value['mean'])}, Mean value: {value['mean']}")
                if 'std' in value:
                    print(f"Std type: {type(value['std'])}, Std value: {value['std']}")
            else:
                print(f"Value: {value}")
        
        break
    else:
        print(f"No plane statistics found for scene {img_id}")