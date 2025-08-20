"""
Resize PACE 3D models to match target statistics and generate object statistics
"""
import os
import numpy as np
import json
import argparse
import shutil
from tqdm import tqdm
import trimesh
import random

def load_target_statistics():
    """Define target statistics for each object class - SMALLER OBJECTS"""
    target_stats = {
        "s": [0.4, 0.1],
        "m": [0.8, 0.12],
        "l": [1.6, 1.5],
    }
    return target_stats

def map_pace_to_target_classes():
    """Map PACE classes to target object classes"""
    class_mapping = {
        "toy_car": "m",  # Map toy_car to utility_knife as medium object
        "can": "s"        # Map can to tape_measure as small object
    }
    return class_mapping

def get_obj_dimensions(obj_path):
    """Get the dimensions of an OBJ file"""
    try:
        mesh = trimesh.load(obj_path)
        if hasattr(mesh, 'bounds'):
            min_bound, max_bound = mesh.bounds
            dimensions = max_bound - min_bound
            return dimensions
        else:
            return None
    except Exception as e:
        print(f"Error loading {obj_path}: {e}")
        return None

def resize_obj_file(obj_path, target_height, output_path):
    """Resize an OBJ file to match target height"""
    try:
        # Load the mesh
        mesh = trimesh.load(obj_path)
        
        if not hasattr(mesh, 'bounds'):
            print(f"Invalid mesh: {obj_path}")
            return False
        
        # Get current dimensions
        min_bound, max_bound = mesh.bounds
        current_dimensions = max_bound - min_bound
        current_height = current_dimensions[2]  # Z dimension is height
        
        if current_height <= 0:
            print(f"Invalid height {current_height} for {obj_path}")
            return False
        
        # Calculate scale factor
        scale_factor = target_height / current_height
        
        # Apply scaling
        mesh.apply_scale(scale_factor)
        
        # Save the resized mesh
        mesh.export(output_path)
        
        return True
        
    except Exception as e:
        print(f"Error resizing {obj_path}: {e}")
        return False

def copy_related_files(src_dir, dst_dir, obj_name):
    """Copy material and texture files along with OBJ"""
    # List of file extensions to copy
    extensions = ['.mtl', '.png', '.jpg', '.jpeg', '.tga', '.bmp']
    
    for ext in extensions:
        src_file = os.path.join(src_dir, f"material{ext}")
        dst_file = os.path.join(dst_dir, f"material{ext}")
        
        if os.path.exists(src_file):
            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"Warning: Could not copy {src_file}: {e}")
        
        # Also try with material_0 prefix
        src_file = os.path.join(src_dir, f"material_0{ext}")
        dst_file = os.path.join(dst_dir, f"material_0{ext}")
        
        if os.path.exists(src_file):
            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"Warning: Could not copy {src_file}: {e}")

def resize_pace_models(pace_base_dir, output_base_dir, target_stats, class_mapping):
    """Resize all models to match target statistics"""
    
    splits = ['train', 'val', 'test']
    
    # Track statistics for each class
    resized_stats = {}
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Load the objaverse.json for this split
        json_path = os.path.join(pace_base_dir, split, 'objaverse.json')
        
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping {split}")
            continue
        
        with open(json_path, 'r') as f:
            split_data = json.load(f)
        
        # Create output directory for this split
        output_split_dir = os.path.join(output_base_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)
        
        # Process each category
        for pace_class, obj_list in split_data.items():
            print(f"  Processing {pace_class} ({len(obj_list)} objects)...")
            
            # Map PACE class to target class
            target_class = class_mapping.get(pace_class, pace_class)
            
            if target_class not in target_stats:
                print(f"    Warning: No target stats for {target_class}, skipping...")
                continue
            
            target_mean, target_std = target_stats[target_class]
            
            # Initialize statistics tracking
            if target_class not in resized_stats:
                resized_stats[target_class] = []
            
            # Process each object
            for obj_name in tqdm(obj_list, desc=f"Resizing {pace_class}"):
                # Source paths
                src_obj_dir = os.path.join(pace_base_dir, split, obj_name)
                src_obj_path = os.path.join(src_obj_dir, f"{obj_name}.obj")
                
                # Output paths
                dst_obj_dir = os.path.join(output_split_dir, obj_name)
                os.makedirs(dst_obj_dir, exist_ok=True)
                dst_obj_path = os.path.join(dst_obj_dir, f"{obj_name}.obj")
                
                if not os.path.exists(src_obj_path):
                    print(f"    Warning: {src_obj_path} not found")
                    continue
                
                # Generate target height with some variation
                target_height = np.random.normal(target_mean, target_std)
                target_height = max(target_height, target_mean * 0.5)  # Minimum 50% of mean
                target_height = min(target_height, target_mean * 1.5)  # Maximum 150% of mean
                
                # Resize the OBJ file
                success = resize_obj_file(src_obj_path, target_height, dst_obj_path)
                
                if success:
                    # Copy related files (materials, textures)
                    copy_related_files(src_obj_dir, dst_obj_dir, obj_name)
                    
                    # Record the actual resized height for statistics
                    actual_dims = get_obj_dimensions(dst_obj_path)
                    if actual_dims is not None:
                        actual_height = actual_dims[2]
                        resized_stats[target_class].append(actual_height)
                else:
                    print(f"    Failed to resize {obj_name}")
        
        # Create new objaverse.json for resized split
        output_json = os.path.join(output_split_dir, 'objaverse.json')
        with open(output_json, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"  Completed {split} split")
    
    return resized_stats

def generate_statistics_file(resized_stats, output_path):
    """Generate statistics file in the same format as the original script"""
    
    statistics = {}
    
    for class_name, heights in resized_stats.items():
        if len(heights) > 0:
            statistics[class_name] = [
                float(np.mean(heights)),  # Mean height
                float(np.std(heights))    # Standard deviation
            ]
            print(f"{class_name}: mean={np.mean(heights):.3f}, std={np.std(heights):.3f}, count={len(heights)}")
        else:
            print(f"{class_name}: No data")
    
    # Save statistics
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=4)
    
    print(f"\nStatistics saved to: {output_path}")
    return statistics

def verify_resized_models(output_base_dir, expected_stats):
    """Verify that the resized models meet the target statistics"""
    print("\n" + "="*50)
    print("VERIFICATION REPORT")
    print("="*50)
    
    splits = ['train', 'val', 'test']
    all_verification_stats = {}
    
    for split in splits:
        split_dir = os.path.join(output_base_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        json_path = os.path.join(split_dir, 'objaverse.json')
        if not os.path.exists(json_path):
            continue
        
        with open(json_path, 'r') as f:
            split_data = json.load(f)
        
        print(f"\n{split.upper()} Split:")
        
        for pace_class, obj_list in split_data.items():
            heights = []
            
            for obj_name in obj_list:
                obj_path = os.path.join(split_dir, obj_name, f"{obj_name}.obj")
                if os.path.exists(obj_path):
                    dims = get_obj_dimensions(obj_path)
                    if dims is not None:
                        heights.append(dims[2])
            
            if heights:
                mean_height = np.mean(heights)
                std_height = np.std(heights)
                
                # Map to target class for comparison
                class_mapping = map_pace_to_target_classes()
                target_class = class_mapping.get(pace_class, pace_class)
                
                if target_class in expected_stats:
                    target_mean, target_std = expected_stats[target_class]
                    mean_diff = abs(mean_height - target_mean)
                    std_diff = abs(std_height - target_std)
                    
                    status = "✅" if mean_diff < 0.2 and std_diff < 0.2 else "⚠️"
                    
                    print(f"  {status} {pace_class} -> {target_class}:")
                    print(f"    Target: {target_mean:.3f} ± {target_std:.3f}")
                    print(f"    Actual: {mean_height:.3f} ± {std_height:.3f}")
                    print(f"    Count: {len(heights)}")
                
                # Track for overall statistics
                if target_class not in all_verification_stats:
                    all_verification_stats[target_class] = []
                all_verification_stats[target_class].extend(heights)

def main():
    parser = argparse.ArgumentParser(description='Resize 3D models to match target statistics')
    parser.add_argument('--input_dir', 
                       default='/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/Pace/obj',
                       help='Input directory containing train/val/test splits')
    parser.add_argument('--output_dir',
                       default='/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/Pace/obj_resized', 
                       help='Output directory for resized models')
    parser.add_argument('--stats_output',
                       default='/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/PACE_objects_statistic.json',
                       help='Output path for statistics JSON file')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify existing resized models without resizing')
    
    args = parser.parse_args()
    
    print("PACE Model Resizer and Statistics Generator")
    print("=" * 50)
    
    # Load target statistics
    target_stats = load_target_statistics()
    class_mapping = map_pace_to_target_classes()
    
    print("Target Statistics:")
    for class_name, (mean, std) in target_stats.items():
        print(f"  {class_name}: {mean:.1f} ± {std:.1f}")
    
    print("\nClass Mapping:")
    for pace_class, target_class in class_mapping.items():
        print(f"  {pace_class} -> {target_class}")
    
    if args.verify_only:
        # Only verify existing models
        verify_resized_models(args.output_dir, target_stats)
    else:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"\nInput directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        
        # Resize models
        resized_stats = resize_pace_models(args.input_dir, args.output_dir, target_stats, class_mapping)
        
        # Generate statistics file
        generate_statistics_file(resized_stats, args.stats_output)
        
        # Verify the results
        verify_resized_models(args.output_dir, target_stats)
        
        print(f"\n🎉 Resizing complete!")
        print(f"📁 Resized models: {args.output_dir}")
        print(f"📊 Statistics file: {args.stats_output}")

if __name__ == '__main__':
    main()