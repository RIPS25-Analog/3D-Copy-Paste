"""
Convert PACE PLY files to OBJ format and organize them in the desired structure
"""
import trimesh
from tqdm import tqdm
from pathlib import Path
import os
import json
import argparse
import shutil
import glob


def organize_pace_data(source_dir, output_base_dir):
    """
    Organize PACE data files into train split for toy_car, can, and distractor
    """
    # Define the instance splits
    data_splits = {
        'toy_car': {
            'train': {459, 460, 467, 468}
        },
        'can': {
            'train': {74, 57, 58}
        },
        'distractor': {
            'train': {56, 82, 87, 101, 153, 207, 228, 229, 249, 257, 286, 317, 338, 361, 404, 410, 415, 434, 435, 436, 528, 543, 635, 636}
        }
    }

    # Create output directories
    for category in data_splits.keys():
        output_dir = os.path.join(output_base_dir, 'train', category)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    # Process each category and split
    for category, splits in data_splits.items():
        print(f"\nProcessing {category}...")
        for split_name, instance_ids in splits.items():
            print(f"  {split_name} split: {len(instance_ids)} instances")
            output_dir = os.path.join(output_base_dir, split_name, category)
            for instance_id in tqdm(instance_ids, desc=f"{category} {split_name}"):
                instance_str = f"{instance_id:06d}"
                ply_source = os.path.join(source_dir, f"obj_{instance_str}.ply")
                png_source = os.path.join(source_dir, f"obj_{instance_str}.png")
                ply_dest = os.path.join(output_dir, f"obj_{instance_str}.ply")
                png_dest = os.path.join(output_dir, f"obj_{instance_str}.png")
                if os.path.exists(ply_source):
                    shutil.copy2(ply_source, ply_dest)
                    print(f"    Copied: {ply_source} -> {ply_dest}")
                else:
                    print(f"    WARNING: PLY file not found: {ply_source}")
                if os.path.exists(png_source):
                    shutil.copy2(png_source, png_dest)
                    print(f"    Copied: {png_source} -> {png_dest}")
                else:
                    print(f"    WARNING: PNG file not found: {png_source}")

    # Generate summary
    print("\n" + "="*50)
    print("ORGANIZATION SUMMARY")
    print("="*50)
    total_files = 0
    for category in data_splits.keys():
        split_dir = os.path.join(output_base_dir, 'train', category)
        ply_files = len([f for f in os.listdir(split_dir) if f.endswith('.ply')])
        png_files = len([f for f in os.listdir(split_dir) if f.endswith('.png')])
        total_files += ply_files + png_files
        print(f"train/{category}: {ply_files} PLY files, {png_files} PNG files")
    print(f"\nTotal files organized: {total_files}")
    print(f"Output directory: {output_base_dir}")

def verify_data_integrity(output_base_dir):
    """
    Verify that all expected files are present in the organized structure
    """
    print("\n" + "="*50)
    print("DATA INTEGRITY CHECK")
    print("="*50)
    expected_counts = {
        'toy_car': {'train': 4},
        'can': {'train': 3},
        'distractor': {'train': 24}
    }
    all_good = True
    for category, splits in expected_counts.items():
        for split_name, expected_count in splits.items():
            split_dir = os.path.join(output_base_dir, split_name, category)
            if not os.path.exists(split_dir):
                print(f"❌ MISSING: {split_dir}")
                all_good = False
                continue
            ply_files = len([f for f in os.listdir(split_dir) if f.endswith('.ply')])
            png_files = len([f for f in os.listdir(split_dir) if f.endswith('.png')])
            if ply_files == expected_count and png_files == expected_count:
                print(f"✅ {split_name}/{category}: {ply_files} PLY + {png_files} PNG files")
            else:
                print(f"❌ {split_name}/{category}: Expected {expected_count}, got {ply_files} PLY + {png_files} PNG")
                all_good = False
    if all_good:
        print("\n🎉 All data organized successfully!")
    else:
        print("\n⚠️  Some issues found. Please check the warnings above.")
    return all_good

def create_dataset_info(output_base_dir):
    """
    Create a JSON file with dataset information
    """
    import json
    data_splits = {
        'toy_car': {'train': [459, 460, 467, 468]},
        'can': {'train': [74, 57, 58]},
        'distractor': {'train': [56, 82, 87, 101, 153, 207, 228, 229, 249, 257, 286, 317, 338, 361, 404, 410, 415, 434, 435, 436, 528, 543, 635, 636]}
    }
    dataset_info = {
        "dataset_name": "PACE",
        "categories": list(data_splits.keys()),
        "splits": ["train"],
        "file_types": ["ply", "png"],
        "instance_splits": data_splits,
        "total_instances": {
            category: {
                split: len(instances)
                for split, instances in splits.items()
            }
            for category, splits in data_splits.items()
        }
    }
    info_file = os.path.join(output_base_dir, "dataset_info.json")
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"\nDataset info saved to: {info_file}")
def convert_specific_split(source_dir, base_data_dir, target_split, target_category=None):
    """
    Convert PLY files for a specific split and optionally specific category.
    Output directory is fixed to /home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/Pace/train_obj
    """
    import shutil
    import os
    from tqdm import tqdm

    data_splits = {
        'toy_car': {
            'train': {459, 460, 467, 468}
        },
        'can': {
            'train': {74, 57, 58}
        },
        'distractor': {
            'train': {56, 82, 87, 101, 153, 207, 228, 229, 249, 257, 286, 317, 338, 361, 404, 410, 415, 434, 435, 436, 528, 543, 635, 636}
        }
    }

    obj_output_dir = "/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/Pace/train_obj"
    split_output_dir = os.path.join(obj_output_dir, target_split)
    os.makedirs(split_output_dir, exist_ok=True)

    categories_to_process = [target_category] if target_category else list(data_splits.keys())
    converted_data = {}

    for category in categories_to_process:
        if category not in data_splits:
            print(f"Unknown category: {category}")
            continue
        if target_split not in data_splits[category]:
            print(f"Unknown split: {target_split}")
            continue
        print(f"Processing {category} - {target_split}...")
        instance_ids = data_splits[category][target_split]
        converted_objects = []
        for instance_id in tqdm(instance_ids, desc=f"{category} {target_split}"):
            instance_str = f"{instance_id:06d}"
            ply_source = os.path.join(source_dir, f"obj_{instance_str}.ply")
            png_source = os.path.join(source_dir, f"obj_{instance_str}.png")
            obj_folder = os.path.join(split_output_dir, category)
            os.makedirs(obj_folder, exist_ok=True)
            ply_dest = os.path.join(obj_folder, f"obj_{instance_str}.ply")
            png_dest = os.path.join(obj_folder, f"obj_{instance_str}.png")
            if os.path.exists(ply_source):
                shutil.copy2(ply_source, ply_dest)
            if os.path.exists(png_source):
                shutil.copy2(png_source, png_dest)
            converted_objects.append(instance_str)
        converted_data[category] = {target_split: converted_objects}
    return converted_data

def main():
    parser = argparse.ArgumentParser(description='Convert PACE PLY files to OBJ format with organized structure')
    parser.add_argument('--source_dir', 
                       default="/home/data/pace/models",
                       help='Source directory containing PLY and PNG files')
    parser.add_argument('--base_dir', 
                       default="/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P", 
                       help='Base directory for output')
    parser.add_argument('--split', 
                        default='train', 
                        help='Split to convert (only "train" is supported)')
    parser.add_argument('--category', 
                       default=None, 
                       choices=['toy_car', 'can'],
                       help='Specific category to convert (toy_car, can). Only used with --split.')
    
    args = parser.parse_args()
    
    if args.split:
        # Convert specific split
        converted_data = convert_specific_split(args.source_dir, args.base_dir, args.split, args.category)
        print(f"Conversion complete for {args.split} split")
    else:
        # Convert all splits
        pace_dict = convert_pace_ply_to_obj_organized(args.source_dir, args.base_dir)
        print("Conversion complete for all splits")

if __name__ == '__main__':
    main()