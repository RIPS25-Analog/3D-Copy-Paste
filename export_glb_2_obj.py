"""
Convert GLB files to OBJ format and organize them in the desired structure
"""
import trimesh
from tqdm import tqdm
from pathlib import Path
import os
import json
import argparse
import shutil
import glob

def convert_glb_to_obj_organized(base_data_dir):
    """
    Convert GLB files to OBJ format and organize them according to the desired structure
    """
    
    # Paths
    objaverse_data_dir = os.path.join(base_data_dir, "Objaverse_data_resized")
    output_dir = os.path.join(base_data_dir, "objaverse")
    obj_output_dir = os.path.join(output_dir, "obj")
    
    # Create output directories
    os.makedirs(obj_output_dir, exist_ok=True)
    
    # Tool categories to process
    tool_categories = ['drill', 'hammer', 'pliers', 'screwdriver', 'tape_measure', 'wrench']
    
    # Dictionary to store the organized structure for objaverse.json
    objaverse_dict = {}
    
    # Process each tool category
    for category in tool_categories:
        print(f"\nProcessing {category}...")
        category_dir = os.path.join(objaverse_data_dir, category)
        models_dir = os.path.join(category_dir, "models")
        
        if not os.path.exists(models_dir):
            print(f"No models directory found for {category}, skipping...")
            continue
        
        # Find all GLB files in the models directory
        glb_files = glob.glob(os.path.join(models_dir, "*.glb"))
        
        if not glb_files:
            print(f"No GLB files found in {models_dir}")
            continue
        
        objaverse_dict[category] = []
        
        # Process each GLB file
        for i, glb_file in enumerate(tqdm(glb_files, desc=f"Converting {category}")):
            try:
                # Get the base filename without extension
                base_filename = os.path.splitext(os.path.basename(glb_file))[0]
                
                # Create output directory for this object
                obj_folder = os.path.join(obj_output_dir, base_filename)
                os.makedirs(obj_folder, exist_ok=True)
                
                # Output paths
                obj_file = os.path.join(obj_folder, f"{base_filename}.obj")
                
                # Skip if already converted
                if os.path.exists(obj_file):
                    print(f"Already converted: {base_filename}")
                    objaverse_dict[category].append(base_filename)
                    continue
                
                print(f"Converting: {base_filename}")
                
                # Load and convert GLB to OBJ
                mesh = trimesh.load(glb_file)
                
                # Export as OBJ with materials
                mesh.export(
                    file_obj=obj_file,
                    file_type="obj",
                    include_texture=True
                )
                
                # Check if material files were created and rename them appropriately
                # Trimesh sometimes creates materials with default names
                mtl_files = glob.glob(os.path.join(obj_folder, "*.mtl"))
                png_files = glob.glob(os.path.join(obj_folder, "*.png"))
                jpg_files = glob.glob(os.path.join(obj_folder, "*.jpg"))
                
                # Rename material files to standard names
                if mtl_files:
                    standard_mtl = os.path.join(obj_folder, "material.mtl")
                    if mtl_files[0] != standard_mtl:
                        shutil.move(mtl_files[0], standard_mtl)
                
                # Rename texture files to standard names
                if png_files:
                    standard_png = os.path.join(obj_folder, "material.png")
                    if png_files[0] != standard_png:
                        shutil.move(png_files[0], standard_png)
                elif jpg_files:
                    standard_jpg = os.path.join(obj_folder, "material.jpg")
                    if jpg_files[0] != standard_jpg:
                        shutil.move(jpg_files[0], standard_jpg)
                
                # Add to objaverse dictionary
                objaverse_dict[category].append(base_filename)
                
                print(f"Successfully converted: {base_filename}")
                
            except Exception as e:
                print(f"Error converting {glb_file}: {str(e)}")
                # Log error
                error_log = os.path.join(output_dir, "conversion_errors.txt")
                with open(error_log, "a") as f:
                    f.write(f"Error converting {glb_file}: {str(e)}\n")
                continue
    
    # Save objaverse.json
    objaverse_json_path = os.path.join(output_dir, "objaverse.json")
    with open(objaverse_json_path, "w") as f:
        json.dump(objaverse_dict, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"objaverse.json saved to: {objaverse_json_path}")
    print(f"OBJ files saved to: {obj_output_dir}")
    
    # Print summary
    total_objects = sum(len(objects) for objects in objaverse_dict.values())
    print(f"\nSummary:")
    for category, objects in objaverse_dict.items():
        print(f"  {category}: {len(objects)} objects")
    print(f"  Total: {total_objects} objects converted")
    
    return objaverse_dict

def convert_specific_category(base_data_dir, category):
    """
    Convert GLB files for a specific category only
    """
    
    # Paths
    objaverse_data_dir = os.path.join(base_data_dir, "Objaverse_data_resized")
    output_dir = os.path.join(base_data_dir, "objaverse")
    obj_output_dir = os.path.join(output_dir, "obj")
    
    # Create output directories
    os.makedirs(obj_output_dir, exist_ok=True)
    
    print(f"Processing {category}...")
    category_dir = os.path.join(objaverse_data_dir, category)
    models_dir = os.path.join(category_dir, "models")
    
    if not os.path.exists(models_dir):
        print(f"No models directory found for {category}")
        return []
    
    # Find all GLB files
    glb_files = glob.glob(os.path.join(models_dir, "*.glb"))
    
    if not glb_files:
        print(f"No GLB files found in {models_dir}")
        return []
    
    converted_objects = []
    
    # Process each GLB file
    for glb_file in tqdm(glb_files, desc=f"Converting {category}"):
        try:
            # Get the base filename without extension
            base_filename = os.path.splitext(os.path.basename(glb_file))[0]
            
            # Create output directory for this object
            obj_folder = os.path.join(obj_output_dir, base_filename)
            os.makedirs(obj_folder, exist_ok=True)
            
            # Output paths
            obj_file = os.path.join(obj_folder, f"{base_filename}.obj")
            
            # Skip if already converted
            if os.path.exists(obj_file):
                print(f"Already converted: {base_filename}")
                converted_objects.append(base_filename)
                continue
            
            print(f"Converting: {base_filename}")
            
            # Load and convert GLB to OBJ
            mesh = trimesh.load(glb_file)
            
            # Export as OBJ with materials
            mesh.export(
                file_obj=obj_file,
                file_type="obj",
                include_texture=True
            )
            
            # Handle material files
            mtl_files = glob.glob(os.path.join(obj_folder, "*.mtl"))
            png_files = glob.glob(os.path.join(obj_folder, "*.png"))
            jpg_files = glob.glob(os.path.join(obj_folder, "*.jpg"))
            
            # Standardize material file names
            if mtl_files and mtl_files[0] != os.path.join(obj_folder, "material.mtl"):
                shutil.move(mtl_files[0], os.path.join(obj_folder, "material.mtl"))
            
            if png_files and png_files[0] != os.path.join(obj_folder, "material.png"):
                shutil.move(png_files[0], os.path.join(obj_folder, "material.png"))
            elif jpg_files and jpg_files[0] != os.path.join(obj_folder, "material.jpg"):
                shutil.move(jpg_files[0], os.path.join(obj_folder, "material.jpg"))
            
            converted_objects.append(base_filename)
            print(f"Successfully converted: {base_filename}")
            
        except Exception as e:
            print(f"Error converting {glb_file}: {str(e)}")
            continue
    
    # Update objaverse.json
    objaverse_json_path = os.path.join(output_dir, "objaverse.json")
    
    # Load existing objaverse.json if it exists
    objaverse_dict = {}
    if os.path.exists(objaverse_json_path):
        with open(objaverse_json_path, "r") as f:
            objaverse_dict = json.load(f)
    
    # Update with new category
    objaverse_dict[category] = converted_objects
    
    # Save updated objaverse.json
    with open(objaverse_json_path, "w") as f:
        json.dump(objaverse_dict, f, indent=2)
    
    print(f"Converted {len(converted_objects)} {category} objects")
    return converted_objects

def main():
    parser = argparse.ArgumentParser(description='Convert GLB files to OBJ format with organized structure')
    parser.add_argument('--base_dir', 
                       default="/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P", 
                       help='Base directory containing Objaverse_data')
    parser.add_argument('--category', 
                       default=None, 
                       help='Specific category to convert (e.g., hammer, drill). If not specified, converts all categories.')
    
    args = parser.parse_args()
    
    if args.category:
        # Convert specific category
        converted_objects = convert_specific_category(args.base_dir, args.category)
        print(f"Conversion complete for {args.category}: {len(converted_objects)} objects")
    else:
        # Convert all categories
        objaverse_dict = convert_glb_to_obj_organized(args.base_dir)
        print("Conversion complete for all categories")

if __name__ == '__main__':
    main()