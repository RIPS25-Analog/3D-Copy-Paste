# Enhanced 3D Copy-Paste with Adaptive Scaling and Multi-Surface Support - SILENT VERSION WITH MATERIAL FIX
import bpy
from mathutils import *
from math import *
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import json
import os
from shapely.geometry import Polygon
import cv2
import argparse
import time
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Suppress Blender output
def suppress_blender_output():
    """Suppress all Blender console output (safe for all Blender versions)"""
    try:
        # Set Blender logging to only show errors
        import logging
        logging.getLogger('bpy').setLevel(logging.ERROR)
        
        # Suppress render progress output
        bpy.app.debug_value = 0
        
        # Set preferences to minimize output (safe version)
        try:
            if hasattr(bpy.context, 'preferences') and hasattr(bpy.context.preferences, 'system'):
                system_prefs = bpy.context.preferences.system
                # Only set attributes that exist
                if hasattr(system_prefs, 'use_scripts_auto_execute'):
                    system_prefs.use_scripts_auto_execute = False
        except:
            pass  # Skip if preferences can't be accessed
            
    except Exception as e:
        # If anything fails, just continue without suppression
        pass

def setup_better_material_rendering():
    """Setup better rendering settings for materials"""
    # Set render engine to Cycles for better material support
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Enable GPU rendering if available
    try:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'
    except:
        pass  # Fallback to CPU if GPU not available
    
    # Set better sampling settings
    bpy.context.scene.cycles.samples = 128  # Reduced for faster rendering
    bpy.context.scene.cycles.preview_samples = 32
    
    # Enable denoising
    bpy.context.scene.cycles.use_denoising = True

def fix_object_materials(obj, obj_dir, obj_name):
    """Fix materials and textures for imported object"""
    import os
    
    if not obj.data or not hasattr(obj.data, 'materials'):
        return
    
    # Common texture file extensions
    texture_extensions = ['.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tiff']
    
    for mat_slot in obj.material_slots:
        if mat_slot.material is None:
            continue
            
        material = mat_slot.material
        
        # Enable nodes for material
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        # Find or create principled BSDF node
        principled = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
                break
        
        if principled is None:
            principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # Look for texture files
        texture_found = False
        for ext in texture_extensions:
            # Try different naming conventions
            texture_patterns = [
                f"{obj_name}{ext}",
                f"{obj_name}_diffuse{ext}",
                f"{obj_name}_albedo{ext}",
                f"{obj_name}_color{ext}",
                f"{obj_name}_basecolor{ext}",
                f"texture{ext}",
                f"diffuse{ext}",
                f"material{ext}"
            ]
            
            for pattern in texture_patterns:
                texture_path = os.path.join(obj_dir, pattern)
                if os.path.exists(texture_path):
                    try:
                        # Load the image
                        image = bpy.data.images.load(texture_path)
                        
                        # Create texture node
                        tex_node = nodes.new(type='ShaderNodeTexImage')
                        tex_node.image = image
                        
                        # Connect to principled BSDF
                        links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                        
                        # If the image has alpha, connect it
                        if image.channels == 4:  # RGBA
                            links.new(tex_node.outputs['Alpha'], principled.inputs['Alpha'])
                            material.blend_method = 'BLEND'
                        
                        texture_found = True
                        break
                        
                    except Exception as e:
                        continue
            
            if texture_found:
                break
        
        # If no texture found, try to parse MTL file manually
        if not texture_found:
            try:
                load_material_from_mtl(material, obj_dir, obj_name)
            except Exception as e:
                # Set a default color if no texture is found
                if principled:
                    # Use a neutral color instead of bright default
                    principled.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)  # Light gray

def load_material_from_mtl(material, obj_dir, obj_name):
    """Parse MTL file and load textures manually"""
    import os
    
    mtl_path = os.path.join(obj_dir, obj_name + '.mtl')
    if not os.path.exists(mtl_path):
        return
    
    try:
        with open(mtl_path, 'r') as f:
            lines = f.readlines()
        
        current_material = None
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        
        # Find principled BSDF
        principled = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
                break
        
        if principled is None:
            return
        
        for line in lines:
            line = line.strip()
            if line.startswith('newmtl '):
                current_material = line[7:]
            elif line.startswith('map_Kd ') or line.startswith('map_Ka '):
                # Diffuse texture
                texture_file = line.split(' ', 1)[1].strip()
                texture_path = os.path.join(obj_dir, texture_file)
                
                if os.path.exists(texture_path):
                    try:
                        image = bpy.data.images.load(texture_path)
                        tex_node = nodes.new(type='ShaderNodeTexImage')
                        tex_node.image = image
                        links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                        
                        if image.channels == 4:
                            links.new(tex_node.outputs['Alpha'], principled.inputs['Alpha'])
                            material.blend_method = 'BLEND'
                        
                        return
                    except Exception as e:
                        pass
            
            elif line.startswith('Kd '):
                # Diffuse color
                try:
                    colors = line.split()[1:4]
                    if len(colors) == 3:
                        r, g, b = map(float, colors)
                        principled.inputs['Base Color'].default_value = (r, g, b, 1.0)
                except:
                    pass
                    
    except Exception as e:
        pass

def load_insert_object(filepath):
    """Load OBJ file with proper material and texture support"""
    import os
    
    # Get the directory containing the .obj file
    obj_dir = os.path.dirname(filepath)
    obj_name = os.path.splitext(os.path.basename(filepath))[0]
    
    # Suppress import messages
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        # Import with material support enabled
        bpy.ops.import_scene.obj(
            filepath=filepath,
            use_edges=True,
            use_smooth_groups=True,
            use_split_objects=False,
            use_split_groups=False,
            use_groups_as_vgroups=False,
            use_image_search=True,  # Search for images in subdirectories
            split_mode='ON',
            global_clamp_size=0,
            axis_forward='-Z',
            axis_up='Y'
        )
    
    # Get the imported object
    inobject = bpy.context.active_object
    if inobject is None and bpy.context.selected_objects:
        inobject = bpy.context.selected_objects[-1]
    if inobject is None:
        raise RuntimeError("No object was imported")
    
    # Ensure the object has a valid name
    if inobject.name is None or inobject.name == "":
        inobject.name = "ImportedObject"
    
    # Try to fix materials and textures
    fix_object_materials(inobject, obj_dir, obj_name)
    
    # Create bounding box
    bpy.ops.mesh.primitive_cube_add()
    bbox = bpy.context.active_object # Get the cube directly
    if bbox is None:
        raise RuntimeError("Failed to create bounding box cube")
    # Set bbox properties
    bbox.name = f"{inobject.name}_bbox"
    bbox.location = inobject.location.copy()
    bbox.dimensions = inobject.dimensions.copy()
    bbox.hide_render = True # Hide the bounding box in render
    # Reselect the original object
    bpy.context.view_layer.objects.active = inobject
    return inobject, bbox

def select_inserted_object_class(objaverse_object_name_list_x, ori_cate_list, insertion_mode):
    '''
    Randomly select one class given class list and insertion_mode
    insertion_mode: random or context
    '''
    if insertion_mode == 'random':
        samples = random.choice(objaverse_object_name_list_x)
        return samples
    elif insertion_mode == 'context':
        candidate_class_list = []
        for cate in ori_cate_list:
            if cate in objaverse_object_name_list_x:
                candidate_class_list.append(cate)
        if candidate_class_list == []: # empty
            candidate_class_list = objaverse_object_name_list_x
        samples = random.choice(candidate_class_list)
        return samples

def iou_rotated_rectangles(corners1, corners2):
    """
    Compute the intersection over union (IOU) of two rotated rectangles
    """
    area1 = polygon_area(corners1)
    area2 = polygon_area(corners2)
    intersection_corners = polygon_intersection(corners1, corners2)
    intersection_area = polygon_area(intersection_corners)
    iou = intersection_area / (area1 + area2 - intersection_area)
    return iou

def polygon_area(corners):
    """
    Calculate the signed area of a polygon using the shoelace formula
    """
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def polygon_intersection(corners1, corners2):
    """
    Calculate the intersection of two polygons
    """
    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)
    intersection_poly = poly1.intersection(poly2)
    if intersection_poly.is_empty:
        return []
    intersection_corners = list(intersection_poly.exterior.coords)[:-1]
    return intersection_corners

def get_rotated_rect_corners(center, width, height, angle):
    """
    Calculate the corner positions of a rectangle after rotation
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    corners = [(center[0] + width / 2 * cos_a - height / 2 * sin_a, center[1] + width / 2 * sin_a + height / 2 * cos_a),
               (center[0] - width / 2 * cos_a - height / 2 * sin_a, center[1] - width / 2 * sin_a + height / 2 * cos_a),
               (center[0] - width / 2 * cos_a + height / 2 * sin_a, center[1] - width / 2 * sin_a - height / 2 * cos_a),
               (center[0] + width / 2 * cos_a + height / 2 * sin_a, center[1] + width / 2 * sin_a - height / 2 * cos_a)]
    return corners

def check_collision(ori_GT_object_info_dict, sample_bbox_info):
    collision_info = {}
    
    # FIXED: Calculate bbox_corners outside the loop since it doesn't depend on loop iteration
    bbox_center_X, bbox_center_Y, bbox_size_x, bbox_size_y, bbox_rotation = sample_bbox_info
    bbox_corners = get_rotated_rect_corners(center=(bbox_center_X, bbox_center_Y), width=bbox_size_x * 2,
                                          height=bbox_size_y * 2, angle=bbox_rotation)
    
    for obj_name, info in ori_GT_object_info_dict.items():
        collision_info[obj_name] = {}
        center_X = info[4]
        center_Y = info[5]
        size_x = info[8] # length
        size_y = info[7] # width
        angle = np.arctan2(info[11], info[10]) * 180 / pi
        corners = get_rotated_rect_corners(center=(center_X, center_Y), width=size_x * 2, height=size_y * 2,
                                         angle=angle)
        
        collision_info[obj_name]['iou'] = iou_rotated_rectangles(corners, bbox_corners)
        collision_info[obj_name]['corners'] = corners
    
    # Now bbox_corners is always defined
    collision_info['bbox_corners'] = bbox_corners
    return collision_info

def get_view_projection_matrices(cam):
    scene = bpy.context.scene
    render = scene.render
    aspect_ratio = render.resolution_x / render.resolution_y
    cam_data = cam.data
    # Create the view matrix
    mat_view = cam.matrix_world.inverted()
    # Create the projection matrix
    if cam_data.type == 'PERSP':
        # Perspective camera
        fovy = cam_data.angle_y
        near = cam_data.clip_start
        far = cam_data.clip_end
        top = near * (fovy / 2)
        right = top * aspect_ratio
        mat_proj = Matrix(((near / right, 0, 0, 0),
                          (0, near / top, 0, 0),
                          (0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)),
                          (0, 0, -1, 0)))
    else: # cam_data.type == 'ORTHO'
        # Orthographic camera
        scale_x = cam_data.ortho_scale
        scale_y = cam_data.ortho_scale * aspect_ratio
        near = cam_data.clip_start
        far = cam_data.clip_end
        right = scale_x / 2
        top = scale_y / 2
        mat_proj = Matrix(((1 / right, 0, 0, 0),
                          (0, 1 / top, 0, 0),
                          (0, 0, -2 / (far - near), -(far + near) / (far - near)),
                          (0, 0, 0, 1)))
    return mat_view, mat_proj

def world_to_screen_coords(world_pos, cam):
    """Quick fix using Blender's built-in projection"""
    from bpy_extras.object_utils import world_to_camera_view
    scene = bpy.context.scene
    # Use Blender's built-in world to camera conversion
    co_2d = world_to_camera_view(scene, cam, world_pos)
    # Convert to pixel coordinates
    render = scene.render
    screen_x = co_2d.x * render.resolution_x
    screen_y = (1 - co_2d.y) * render.resolution_y # Flip Y axis
    return screen_x, screen_y

def load_environment_map(filepath, rotation_degrees, intensity):
    try:
        # Check if file exists first
        if not os.path.exists(filepath):
            return
        
        # Load the image
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            bpy.ops.image.open(filepath=filepath)
        
        # Get the image object
        env_map = bpy.data.images[os.path.basename(filepath)]
        
        # Create a new world if necessary
        if not bpy.data.worlds.get("Environment_World"):
            bpy.data.worlds.new("Environment_World")
        # Set the new world as the active world
        world = bpy.data.worlds["Environment_World"]
        bpy.context.scene.world = world
        # Use nodes for the world
        world.use_nodes = True
        # Clear existing nodes
        nodes = world.node_tree.nodes
        nodes.clear()
        # Create the necessary nodes
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        background_node = nodes.new(type='ShaderNodeBackground')
        environment_texture_node = nodes.new(type='ShaderNodeTexEnvironment')
        mapping_node = nodes.new(type='ShaderNodeMapping')
        texture_coord_node = nodes.new(type='ShaderNodeTexCoord')
        # Set the environment texture
        environment_texture_node.image = env_map
        # Set rotation
        mapping_node.inputs[2].default_value[2] = rotation_degrees * 0.0174533 # Convert degrees to radians
        # control intensity of the environment map (MUCH DARKER LIGHTING)
        background_node.inputs[1].default_value = intensity * 0.3  # 70% darker than before
        # Connect the nodes
        links = world.node_tree.links
        links.new(texture_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], environment_texture_node.inputs['Vector'])
        links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])
        links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
        
    except Exception as e:
        setup_default_environment(intensity)

def select_placement_surface(scene_id, root_path, surface_selection_mode='random'):
    """
    Select a surface for object placement
    surface_selection_mode: 'random', 'floor_only', 'tables_only', or 'weighted'
    Returns: surface_info dict with all necessary placement information
    """
    # First try enhanced plane detection
    enhanced_plane_path = os.path.join(
        root_path,
        'plane/{}/enhanced_plane_statistics.json'.format(scene_id)
    )
    if os.path.exists(enhanced_plane_path):
        with open(enhanced_plane_path, 'r') as fh:
            plane_data = json.load(fh)
        # Collect all available surfaces
        available_surfaces = []
        # Add floor if available
        if 'floor_id' in plane_data and plane_data['floor_id'] in plane_data['horizontal_planes']:
            floor_id = plane_data['floor_id']
            available_surfaces.append({
                'type': 'floor',
                'stats': plane_data['horizontal_planes'][floor_id],
                'id': floor_id
            })
        # Add all other surfaces
        for surface_type in ['table', 'desk', 'counter', 'surface']:
            if surface_type in plane_data.get('surfaces', {}):
                for surface in plane_data['surfaces'][surface_type]:
                    available_surfaces.append({
                        'type': surface_type,
                        'stats': surface['stats'],
                        'id': surface['id']
                    })
        if available_surfaces:
            # Select surface based on mode
            if surface_selection_mode == 'random':
                return random.choice(available_surfaces)
            elif surface_selection_mode == 'weighted':
                areas = [surf['stats']['area'] for surf in available_surfaces]
                total_area = sum(areas)
                weights = [area / total_area for area in areas]
                return random.choices(available_surfaces, weights=weights)[0]
            elif surface_selection_mode == 'floor_only':
                for surf in available_surfaces:
                    if surf['type'] == 'floor':
                        return surf
            elif surface_selection_mode == 'tables_only':
                non_floor = [s for s in available_surfaces if s['type'] != 'floor']
                if non_floor:
                    return random.choice(non_floor)
    
    # Fallback to original floor-only detection
    floor_plane_path = os.path.join(
        root_path,
        'plane/{}/floor_plane_statistics_noceiling_threshold0.04.json'.format(scene_id)
    )
    if os.path.exists(floor_plane_path):
        with open(floor_plane_path, 'r') as fh:
            plane_data = json.load(fh)
        return {
            'type': 'floor',
            'stats': plane_data['floor'],
            'id': plane_data.get('floor_id', 'floor')
        }
    return None

def calculate_adaptive_object_scale(surface_info, bbox_dimensions, insertion_size_statistics,
                                  insert_class, min_coverage=0.15, max_coverage=0.5):
    """
    Calculate appropriate object scale based on surface size
    """
    surface_area = surface_info['stats']['area']
    surface_width = surface_info['stats']['max'][0] - surface_info['stats']['min'][0]
    surface_length = surface_info['stats']['max'][1] - surface_info['stats']['min'][1]
    
    # Get reference height from statistics
    if insert_class in insertion_size_statistics:
        size_z = random.gauss(
            mu=insertion_size_statistics[insert_class][0],
            sigma=insertion_size_statistics[insert_class][1]
        )
    else:
        size_z = 0.3 # Default height if class not in statistics
    
    reference_object_height = size_z * 2
    obj_shrink_factor = bbox_dimensions[2] / reference_object_height
    
    # Object base dimensions after initial scaling
    scaled_width = bbox_dimensions[0] / obj_shrink_factor
    scaled_length = bbox_dimensions[1] / obj_shrink_factor
    scaled_base_area = scaled_width * scaled_length
    
    # Calculate scale factors based on coverage
    min_scale = np.sqrt(surface_area * min_coverage / scaled_base_area)
    max_scale = np.sqrt(surface_area * max_coverage / scaled_base_area)
    
    # Check dimension constraints
    width_scale = surface_width * 0.8 / scaled_width # 80% of surface width max
    length_scale = surface_length * 0.8 / scaled_length
    
    # Final scale is the minimum of all constraints
    max_allowed_scale = min(max_scale, width_scale, length_scale)
    
    # Random scale between min and max
    if max_allowed_scale > min_scale:
        additional_scale = random.uniform(min_scale, max_allowed_scale)
    else:
        additional_scale = max_allowed_scale * 0.9
    
    # Ensure minimum visibility
    min_visible_scale = 0.5
    additional_scale = max(additional_scale, min_visible_scale)
    
    return obj_shrink_factor, additional_scale

def check_object_in_camera_view(obj_location, camera):
    """Check if object is within camera's field of view"""
    from bpy_extras.object_utils import world_to_camera_view
    scene = bpy.context.scene
    co_2d = world_to_camera_view(scene, camera, obj_location)
    return 0 <= co_2d.x <= 1 and 0 <= co_2d.y <= 1 and co_2d.z > 0

def add_debug_lighting(object_location):
    """Add lighting to illuminate the object (MUCH DARKER LIGHTING)"""
    # Add a sun light (much reduced energy)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun_light = bpy.context.active_object
    sun_light.data.energy = 2.0  # Reduced from 5.0 to 2.0
    sun_light.rotation_euler = (0.5, 0.5, 0)
    
    # Add area light near object (much reduced energy)
    bpy.ops.object.light_add(type='AREA', location=(object_location[0], object_location[1], object_location[2] + 2))
    area_light = bpy.context.active_object
    area_light.data.energy = 80.0  # Reduced from 200.0 to 80.0
    area_light.data.size = 2.0

def setup_default_environment(strength=1.0):
    """Set up a simple default environment when HDR file is missing"""
    try:
        world = bpy.context.scene.world
        if world is None:
            return
        
        world.use_nodes = True
        node_tree = world.node_tree
        node_tree.nodes.clear()
        
        # Simple background shader with default color (MUCH DARKER)
        background_node = node_tree.nodes.new('ShaderNodeBackground')
        background_node.inputs['Color'].default_value = (0.01, 0.01, 0.01, 1.0) # Much darker gray
        background_node.inputs['Strength'].default_value = strength * 0.3  # 70% darker than before
        
        output_node = node_tree.nodes.new('ShaderNodeOutputWorld')
        node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
        
    except Exception as e:
        pass

def get_2d_bounding_box(obj, cam, scene):
    """
    Calculate 2D bounding box of object in screen coordinates
    Returns: (min_x, min_y, max_x, max_y) in pixel coordinates
    """
    # Get object's 8 corner points in world coordinates
    bpy.context.view_layer.update()
    corners_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Project all corners to screen coordinates
    screen_coords = []
    for corner in corners_world:
        screen_x, screen_y = world_to_screen_coords(corner, cam)
        screen_coords.append((screen_x, screen_y))
    
    # Find bounding box in screen space
    if screen_coords:
        xs = [coord[0] for coord in screen_coords]
        ys = [coord[1] for coord in screen_coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Clamp to image bounds
        width = scene.render.resolution_x
        height = scene.render.resolution_y
        min_x = max(0, min(min_x, width))
        max_x = max(0, min(max_x, width))
        min_y = max(0, min(min_y, height))
        max_y = max(0, min(max_y, height))
        
        return min_x, min_y, max_x, max_y
    
    return None

def convert_to_yolo_format(bbox_coords, image_width, image_height, class_id=0):
    """
    Convert bounding box to YOLOv11 format
    bbox_coords: (min_x, min_y, max_x, max_y) in pixel coordinates
    Returns: (class_id, center_x, center_y, width, height) in normalized coordinates
    """
    if bbox_coords is None:
        return None
    
    min_x, min_y, max_x, max_y = bbox_coords
    
    # Calculate center and dimensions
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    width = max_x - min_x
    height = max_y - min_y
    
    # Normalize to 0-1 range
    center_x_norm = center_x / image_width
    center_y_norm = center_y / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    
    return class_id, center_x_norm, center_y_norm, width_norm, height_norm

def create_class_mapping():
    """Create mapping from class names to YOLO class IDs"""
    # Common object classes - adjust as needed for your dataset
    class_mapping = {
        'wrench': 0,
        'screwdriver': 1,
        'hammer': 2,
        'pliers': 3,
        'knife': 4,
        'scissors': 5,
        'bottle': 6,
        'cup': 7,
        'bowl': 8,
        'plate': 9,
        'book': 10,
        'phone': 11,
        'laptop': 12,
        'mouse': 13,
        'keyboard': 14,
        'monitor': 15,
        'chair': 16,
        'table': 17,
        'bed': 18,
        'sofa': 19,
        # Add more classes as needed
    }
    return class_mapping

def save_yolo_annotation(bbox_data, scene_id, iter, output_dir):
    """Save YOLO format annotation to file"""
    if bbox_data is None:
        return
    
    class_id, center_x, center_y, width, height = bbox_data
    
    # Create annotation string
    annotation_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
    
    # Save to file
    annotation_file = os.path.join(output_dir, f"{scene_id}_{iter}.txt")
    with open(annotation_file, 'w') as f:
        f.write(annotation_line)

def draw_bounding_box_on_image(image_path, bbox_data, class_name, output_path):
    """Draw 2D bounding box annotation on the image"""
    if bbox_data is None:
        return
    
    try:
        import cv2
        import numpy as np
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return
        
        height, width = img.shape[:2]
        class_id, center_x, center_y, bbox_width, bbox_height = bbox_data
        
        # Convert normalized YOLO coordinates to pixel coordinates
        center_x_px = int(center_x * width)
        center_y_px = int(center_y * height) 
        bbox_width_px = int(bbox_width * width)
        bbox_height_px = int(bbox_height * height)
        
        # Calculate top-left and bottom-right corners
        x1 = int(center_x_px - bbox_width_px / 2)
        y1 = int(center_y_px - bbox_height_px / 2)
        x2 = int(center_x_px + bbox_width_px / 2)
        y2 = int(center_y_px + bbox_height_px / 2)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # Draw bounding box rectangle (bright green)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw class label background
        label = f"{class_name} ({class_id})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Position label above the box, or below if too close to top
        label_y = y1 - 10 if y1 > 30 else y2 + 25
        label_x = x1
        
        # Draw label background rectangle (semi-transparent)
        cv2.rectangle(img, (label_x, label_y - label_size[1] - 5), 
                     (label_x + label_size[0] + 5, label_y + 5), (0, 255, 0), -1)
        
        # Draw label text (black on green background)
        cv2.putText(img, label, (label_x + 2, label_y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw center point (small circle)
        cv2.circle(img, (center_x_px, center_y_px), 4, (0, 0, 255), -1)
        
        # Save annotated image
        cv2.imwrite(output_path, img)
        
    except Exception as e:
        pass  # Continue silently if annotation fails

def validate_yolo_bbox(bbox_data, min_size=0.01):
    """Validate YOLO bounding box to ensure it's visible and reasonable"""
    if bbox_data is None:
        return False
    
    class_id, center_x, center_y, width, height = bbox_data
    
    # Check if center is within image bounds (with some margin)
    if not (0.1 <= center_x <= 0.9 and 0.1 <= center_y <= 0.9):
        return False
    
    # Check if size is reasonable
    if width < min_size or height < min_size:
        return False
    
    return True

def force_object_in_view(obj, cam, scene, surface_center, surface_std, surface_bounds):
    """Force object to be positioned within camera view with proper scaling"""
    from bpy_extras.object_utils import world_to_camera_view
    
    max_attempts = 50
    for attempt in range(max_attempts):
        # Try different positions around the surface center
        if attempt < 10:
            # First 10 attempts: close to surface center
            offset_factor = 0.3
        elif attempt < 30:
            # Next 20 attempts: wider search
            offset_factor = 0.7
        else:
            # Final attempts: much wider search
            offset_factor = 1.2
        
        # Sample position with increasing variance
        new_x = surface_center[0] + random.gauss(0, surface_std[0] * offset_factor)
        new_y = surface_center[1] + random.gauss(0, surface_std[1] * offset_factor)
        
        # Clamp to surface bounds
        new_x = np.clip(new_x, surface_bounds['min'][0], surface_bounds['max'][0])
        new_y = np.clip(new_y, surface_bounds['min'][1], surface_bounds['max'][1])
        
        # Calculate Z position (keep object above surface)
        surface_top_z = surface_bounds['max'][2]
        new_z = surface_top_z + 0.3  # 30cm above surface
        
        # Test this position
        test_location = Vector((new_x, new_y, new_z))
        co_2d = world_to_camera_view(scene, cam, test_location)
        
        # Check if position is in good camera view (not too close to edges)
        if 0.2 <= co_2d.x <= 0.8 and 0.2 <= co_2d.y <= 0.8 and co_2d.z > 0:
            obj.location = test_location
            return True
    
    # If all attempts failed, place at surface center with elevated Z
    fallback_location = Vector((surface_center[0], surface_center[1], surface_bounds['max'][2] + 0.5))
    obj.location = fallback_location
    return False

def force_minimum_visible_scale(obj, cam, scene, min_pixel_size=50):  # REDUCED from 150 to 50 pixels
    """Force object to have minimum visible size in pixels (3X SMALLER OBJECTS)"""
    from bpy_extras.object_utils import world_to_camera_view
    
    max_scale = 6.0  # Reduced from 20.0 to 6.0 for smaller objects
    scale_step = 0.3  # Smaller steps
    current_scale = 0.3  # Start much smaller (3x smaller than before)
    
    for attempt in range(30):  # Reduced attempts
        # Apply current scale
        obj.scale = Vector((current_scale, current_scale, current_scale))
        bpy.context.view_layer.update()
        
        # Get 2D bounding box
        bbox_coords = get_2d_bounding_box(obj, cam, scene)
        
        if bbox_coords:
            min_x, min_y, max_x, max_y = bbox_coords
            pixel_width = max_x - min_x
            pixel_height = max_y - min_y
            min_dimension = min(pixel_width, pixel_height)
            
            if min_dimension >= min_pixel_size:
                return current_scale
        
        # Increase scale for next attempt
        current_scale += scale_step
        
        if current_scale > max_scale:
            obj.scale = Vector((max_scale, max_scale, max_scale))
            return max_scale
    
    return current_scale

def main(args):
    '''Prepare'''
    # Initialize timing
    all_times = []
    
    # Suppress Blender output
    suppress_blender_output()
    
    # Setup better material rendering - ADDED THIS LINE
    setup_better_material_rendering()
    
    root_path = args.root_path
    # Load all train scene data
    with open(os.path.join(root_path, 'train_data_idx.txt')) as f:
        train_data_ids_raw = f.readlines()
    train_data_ids = ["{:06d}".format(int(id.split('\n')[0])) for id in train_data_ids_raw]
    
    # Load objaverse pool
    with open('/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/objaverse/objaverse.json') as f:
        objaverse_objects_dict = json.load(f)
    objaverse_object_name_list = list(objaverse_objects_dict.keys())
    obj_root_path = args.obj_root_path
    
    # Insertion hyperparameters
    max_iter = args.max_iter
    random.seed(args.random_seed)
    ilog = args.ilog
    istrength = args.istrength
    insertion_mode = args.insertion_mode
    surface_selection_mode = args.surface_selection_mode
    min_coverage = args.min_coverage
    max_coverage = args.max_coverage
    
    out_folder_name = 'insertion_ilog{}_istren{}_{}_surf{}_cov{}-{}'.format(
        ilog, istrength, insertion_mode, surface_selection_mode,
        int(min_coverage*100), int(max_coverage*100)
    )
    
    # Create output directories INCLUDING YOLO annotations
    os.makedirs(os.path.join(root_path, out_folder_name, 'label'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'inserted_foreground'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'compositional_image'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'annotated_images'), exist_ok=True)  # NEW!
    os.makedirs(os.path.join(root_path, out_folder_name, 'envmap'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'insert_object_log'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'yolo_annotations'), exist_ok=True)
    
    envmap_root = os.path.join(root_path, 'envmap')
    
    '''Go over each scene'''
    for iter in range(max_iter):
        for index, scene_id in enumerate(train_data_ids):
            # START TIMING
            start_time = time.time()
            
            # Select placement surface
            surface_info = select_placement_surface(scene_id, root_path, surface_selection_mode)
            if surface_info is None:
                continue
            
            # Check if output already exists
            if os.path.exists(os.path.join(root_path, out_folder_name, 'compositional_image', '{}_{}.png'.format(scene_id, iter))):
                continue
            
            try:
                # Open the blend file (suppress output)
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    bpy.ops.wm.open_mainfile(filepath=os.path.join(root_path, 'insertion_template.blend'))
                
                # Remove all except camera
                bpy.ops.object.select_all(action='SELECT')
                bpy.data.objects["Camera"].select_set(False)
                bpy.ops.object.delete()
                
                '''1. Background operation - Camera setup'''
                # Load camera information
                with open(os.path.join(root_path, 'calib', '{}.txt'.format(scene_id))) as f:
                    lines = f.readlines()
                
                # Extrinsic
                R_raw = np.array([float(ele) for ele in lines[0].split('\n')[0].split(' ')]).reshape((3, 3)).T
                flip_yz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                R = np.matmul(R_raw, flip_yz)
                matrix_world = np.eye(4)
                matrix_world[:3, :3] = R
                cam = bpy.data.objects['Camera']
                cam.matrix_world = Matrix(matrix_world)
                cam.location = Vector([0, 0, 0])
                
                # Intrinsic
                K = np.array([float(ele) for ele in lines[1].split('\n')[0].split(' ')]).reshape((3, 3)).T
                cx = K[0, 2]
                cy = K[1, 2]
                fx = K[0, 0]
                fy = K[1, 1]
                
                bpy.data.scenes['Scene'].render.resolution_x = int(cx * 2)
                bpy.data.scenes['Scene'].render.resolution_y = int(cy * 2)
                bpy.data.cameras[bpy.data.cameras.keys()[0]].sensor_width = 36
                alpha = bpy.data.scenes['Scene'].render.resolution_x / bpy.data.cameras[bpy.data.cameras.keys()[0]].sensor_width
                bpy.data.cameras[bpy.data.cameras.keys()[0]].lens = fx / alpha
                
                '''2. Insertion operation'''
                # Get surface information for placement
                surface_center_X = surface_info['stats']['mean'][0]
                surface_center_Y = surface_info['stats']['mean'][1]
                surface_center_Z = surface_info['stats']['mean'][2]
                surface_std_X = surface_info['stats']['std'][0]
                surface_std_Y = surface_info['stats']['std'][1]
                
                # Load insertion size statistics
                with open(os.path.join(root_path, 'SUNRGBD_objects_statistic.json'), 'r') as fh:
                    insertion_size_statistics = json.load(fh)
                
                # Get original GT objects
                with open(os.path.join(root_path, 'label', '{}.txt'.format(scene_id))) as f:
                    lines = f.readlines()
                
                ori_GT_object_info_dict = {}
                ori_cate_list = []
                for id, line in enumerate(lines):
                    raw_info_list = line.strip().split()
                    class_name = raw_info_list[0]
                    if class_name not in ori_cate_list:
                        if class_name == 'night_stand':
                            ori_cate_list.append('nightstand')
                        else:
                            ori_cate_list.append(class_name)
                    info_list = [float(ele) for ele in raw_info_list[1:]]
                    ori_GT_object_info_dict[str(id) + '_' + class_name] = info_list
                
                # Select object class
                insert_class = select_inserted_object_class(objaverse_object_name_list, ori_cate_list, insertion_mode)
                
                # Select and load object
                def select_inserted_object(class_name):
                    candidates_dict = objaverse_objects_dict[class_name]
                    uid = random.choice(candidates_dict)
                    obj_path = os.path.join(obj_root_path, uid, uid+'.obj')
                    return obj_path
                
                inserted_object_path = select_inserted_object(insert_class)
                insert_log_dict = {}
                insert_log_dict['insert_class'] = insert_class
                insert_log_dict['inserted_object_path'] = inserted_object_path
                insert_log_dict['surface_type'] = surface_info['type']
                insert_log_dict['surface_id'] = surface_info['id']
                insert_log_dict['surface_area'] = surface_info['stats']['area']
                
                # IMPROVED OBJECT LOADING WITH MATERIALS
                try:
                    inobject, bbox = load_insert_object(filepath=inserted_object_path)
                except Exception as e:
                    print(f"Failed to load object {inserted_object_path}: {e}")
                    continue  # Skip this iteration if object loading fails
                
                # Calculate adaptive scale
                obj_shrink_factor, additional_scale = calculate_adaptive_object_scale(
                    surface_info, bbox.dimensions, insertion_size_statistics,
                    insert_class, min_coverage, max_coverage
                )
                insert_log_dict['obj_shrink_factor'] = obj_shrink_factor
                insert_log_dict['adaptive_scale'] = additional_scale
                
                '''Collision check with adaptive placement'''
                best_parameter = {}
                for i in range(1000):
                    # Use the adaptive scale instead of random uniform
                    sample_scale = additional_scale * random.uniform(0.95, 1.05) # Allow 20% variation
                    
                    # Sample position on the selected surface
                    sample_position_X = random.gauss(surface_center_X, surface_std_X * 0.5)
                    sample_position_Y = random.gauss(surface_center_Y, surface_std_Y * 0.5)
                    
                    # Ensure within surface bounds
                    sample_position_X = np.clip(sample_position_X,
                                              surface_info['stats']['min'][0],
                                              surface_info['stats']['max'][0])
                    sample_position_Y = np.clip(sample_position_Y,
                                              surface_info['stats']['min'][1],
                                              surface_info['stats']['max'][1])
                    sample_rotation = random.uniform(-180, 180)
                    
                    if i == 0:
                        best_parameter['iou'] = 1
                        best_parameter['sample_scale'] = sample_scale
                        best_parameter['sample_position_X'] = sample_position_X
                        best_parameter['sample_position_Y'] = sample_position_Y
                        best_parameter['sample_rotation'] = sample_rotation
                    
                    # Calculate collision
                    sample_obj_shrink_factor = obj_shrink_factor * sample_scale
                    sample_bbox_size = [bbox.dimensions[0] / sample_obj_shrink_factor / 2,
                                      bbox.dimensions[1] / sample_obj_shrink_factor / 2]
                    sample_bbox_info = [sample_position_X, sample_position_Y,
                                      sample_bbox_size[0], sample_bbox_size[1], sample_rotation]
                    collision_info = check_collision(ori_GT_object_info_dict, sample_bbox_info)
                    
                    iou = 0.0
                    for obj in collision_info.keys():
                        if obj != 'bbox_corners':
                            iou += collision_info[obj]['iou']
                    
                    if iou <= 0.01:
                        best_parameter['iou'] = iou
                        best_parameter['sample_scale'] = sample_scale
                        best_parameter['sample_position_X'] = sample_position_X
                        best_parameter['sample_position_Y'] = sample_position_Y
                        best_parameter['sample_rotation'] = sample_rotation
                        break
                    elif iou < best_parameter['iou']:
                        best_parameter['iou'] = iou
                        best_parameter['sample_scale'] = sample_scale
                        best_parameter['sample_position_X'] = sample_position_X
                        best_parameter['sample_position_Y'] = sample_position_Y
                        best_parameter['sample_rotation'] = sample_rotation
                
                # Save insertion information
                insert_log_dict['best_parameter'] = best_parameter
                with open(os.path.join(root_path, out_folder_name, 'insert_object_log', '{}_{}.json'.format(scene_id, iter)), "w") as outfile:
                    json.dump(insert_log_dict, outfile, indent=4)
                print(f"Surface type: {surface_info['type']}")
                print(f"Surface center Z: {surface_center_Z}")
                print(f"Surface max Z: {surface_info['stats']['max'][2]}")
                print(f"Surface min Z: {surface_info['stats']['min'][2]}")
                # Instead of force_object_in_view, use the collision-checked position:
                inobject.location.x = best_parameter['sample_position_X'] 
                inobject.location.y = best_parameter['sample_position_Y']
                inobject.location.z = surface_info['stats']['max'][2] + 0.1
                
                # ===== ENHANCED OBJECT PLACEMENT WITH FORCED VISIBILITY =====
                scene = bpy.context.scene
                cam = bpy.data.objects['Camera']
                
                # Step 1: Force object into camera view
                surface_center = [surface_center_X, surface_center_Y, surface_center_Z]
                surface_std = [surface_std_X, surface_std_Y, 0.1]
                surface_bounds = surface_info['stats']
                
                force_object_in_view(inobject, cam, scene, surface_center, surface_std, surface_bounds)
                
                # Update bbox location to match object
                bbox.location = inobject.location
                
                # Step 2: Apply rotation
                inobject.rotation_mode = 'XYZ'
                inobject.rotation_euler[2] += best_parameter['sample_rotation'] / 180 * pi
                bbox.rotation_euler[2] += best_parameter['sample_rotation'] / 180 * pi
                
                # Step 3: Force minimum visible scale (3X SMALLER)
                final_scale = force_minimum_visible_scale(inobject, cam, scene, min_pixel_size=50)
                
                # Update bbox scale to match
                bbox.scale = inobject.scale
                
                # Step 4: Final positioning adjustments
                bpy.context.view_layer.update()
                
                # Add lighting
                add_debug_lighting(inobject.location)
                
                # Force object to be selected and active for rendering
                bpy.context.view_layer.objects.active = inobject
                inobject.select_set(True)
                
                # ===== GENERATE AND VALIDATE 2D BOUNDING BOX =====
                # Get 2D bounding box coordinates
                bbox_coords = get_2d_bounding_box(inobject, cam, scene)
                yolo_bbox = None  # Initialize for later use
                
                if bbox_coords:
                    # Get class mapping and class ID
                    class_mapping = create_class_mapping()
                    class_id = class_mapping.get(insert_class, 0)
                    
                    # Convert to YOLO format
                    image_width = scene.render.resolution_x
                    image_height = scene.render.resolution_y
                    yolo_bbox = convert_to_yolo_format(bbox_coords, image_width, image_height, class_id)
                    
                    # Validate and save bounding box
                    if validate_yolo_bbox(yolo_bbox, min_size=0.02):
                        yolo_annotations_dir = os.path.join(root_path, out_folder_name, 'yolo_annotations')
                        save_yolo_annotation(yolo_bbox, scene_id, iter, yolo_annotations_dir)
                
                # Save 3D info with corrected scale
                insert_3D_info = {
                    'class': insert_class,
                    'centroid_X': bbox.location[0],
                    'centroid_Y': bbox.location[1],
                    'centroid_Z': bbox.location[2],
                    'width': bbox.dimensions[1] * final_scale / 2,
                    'length': bbox.dimensions[0] * final_scale / 2,
                    'height': bbox.dimensions[2] * final_scale / 2,
                    'heading_angle': best_parameter['sample_rotation'] / 180 * pi,
                    'size': [bbox.dimensions[0] * final_scale,
                            bbox.dimensions[1] * final_scale,
                            bbox.dimensions[2] * final_scale],
                    'surface_type': surface_info['type']
                }
                with open(os.path.join(root_path, out_folder_name, 'label', '{}_{}.json'.format(scene_id, iter)), "w") as outfile:
                    json.dump(insert_3D_info, outfile, indent=4)
                
                # Add shadow plane
                plane_location = bbox.location.copy()
                plane_location[2] = surface_bounds['max'][2] - 0.01
                bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=plane_location)
                plane = bpy.context.active_object
                bpy.data.objects["Plane"].is_shadow_catcher = True
                
                '''3. Dynamic illumination - FIXED HDR ISSUE'''
                backimg = mpimg.imread(os.path.join(root_path, 'image', '{}.jpg'.format(scene_id)))
                object_center_world = Vector((inobject.location[0], inobject.location[1], surface_center_Z))
                screen_x, screen_y = world_to_screen_coords(object_center_world, cam)
                object_center_2d_img = Vector((screen_x, backimg.shape[0] - screen_y))
                object_center_2d_img[0] = np.clip(object_center_2d_img[0], 0, backimg.shape[1])
                object_center_2d_img[1] = np.clip(object_center_2d_img[1], 0, backimg.shape[0])
                
                # Load environment map with improved error handling
                envCol = 160
                envRow = 120
                nh = backimg.shape[0]
                nw = backimg.shape[1]
                if nh < nw:
                    newW = envCol
                    newH = int(float(newW) / float(nw) * nh)
                zoom_ratio = float(nw) / float(newW)
                envmap_index = Vector((object_center_2d_img[1] / zoom_ratio, object_center_2d_img[0] / zoom_ratio))
                
                # Try to find existing environment maps instead of creating HDR
                existing_envmaps = [
                    os.path.join(envmap_root, '{}_envmap0.png'.format(scene_id)),
                    os.path.join(envmap_root, '{}_envmap1.png'.format(scene_id)),
                    os.path.join(envmap_root, '{}.png'.format(scene_id))
                ]
                
                env_map_filepath = None
                for envmap_path in existing_envmaps:
                    if os.path.exists(envmap_path):
                        env_map_filepath = envmap_path
                        break
                
                if env_map_filepath:
                    load_environment_map(env_map_filepath, 180, istrength)
                else:
                    setup_default_environment(istrength)
                
                '''4. Render and save'''
                # Suppress render output
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    bpy.context.scene.render.filepath = os.path.join(root_path, out_folder_name, 'inserted_foreground',
                                                                    '{}_{}.png'.format(scene_id, iter))
                    bpy.ops.render.render(write_still=True)
                
                # Composite images
                filename = os.path.join(root_path, out_folder_name, 'inserted_foreground', '{}_{}.png'.format(scene_id, iter))
                filename1 = os.path.join(root_path, 'image', '{}.jpg'.format(scene_id))
                composite_path = os.path.join(root_path, out_folder_name, 'compositional_image', '{}_{}.png'.format(scene_id, iter))
                
                frontImage = Image.open(filename).convert("RGBA")
                background = Image.open(filename1)
                background.paste(frontImage, (0, 0), frontImage)
                background.save(composite_path, format="png")
                
                # CREATE ANNOTATED VERSION WITH BOUNDING BOX
                if yolo_bbox is not None and validate_yolo_bbox(yolo_bbox, min_size=0.02):
                    annotated_path = os.path.join(root_path, out_folder_name, 'annotated_images', '{}_{}.png'.format(scene_id, iter))
                    draw_bounding_box_on_image(composite_path, yolo_bbox, insert_class, annotated_path)
                
                # Clean up
                bpy.ops.object.select_all(action='SELECT')
                bpy.data.objects["Camera"].select_set(False)
                bpy.ops.object.delete()
                
                # END TIMING
                end_time = time.time()
                processing_time = end_time - start_time
                all_times.append(processing_time)
                
                # Calculate and display average
                current_avg = sum(all_times) / len(all_times)
                print(f"Successfully saved composite image for scene {scene_id} ({processing_time:.2f}s) | Avg: {current_avg:.2f}s")
                
            except Exception as e:
                print(f"Error processing scene {scene_id}: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced 3D Copy-Paste - SILENT VERSION WITH MATERIAL FIX')
    parser.add_argument('--root_path', default="/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/mmdetection3d/data/sunrgbd/sunrgbd_trainval",
                       help='root path for SUN RGB-D data')
    parser.add_argument('--obj_root_path', default="/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/objaverse/obj",
                       help='root path for Objaverse data')
    parser.add_argument('--insertion_mode', default="context",
                       help='random: randomly insert any objects, context: only insert existing category objects')
    parser.add_argument('--surface_selection_mode', default='random',
                       choices=['random', 'weighted', 'floor_only', 'tables_only'],
                       help='How to select placement surface')
    parser.add_argument('--max_iter', type=int, default=3,
                       help='number of inserted objects for each scene')
    parser.add_argument('--random_seed', type=int, default=1,
                       help='random seed for reproduce')
    parser.add_argument('--min_coverage', type=float, default=0.15,
                       help='Minimum surface coverage by object (0.15 = 15%)')
    parser.add_argument('--max_coverage', type=float, default=0.5,
                       help='Maximum surface coverage by object (0.5 = 50%)')
    parser.add_argument('--ilog', type=int, default=2,
                       help='environment map parameter')
    parser.add_argument('--istrength', type=int, default=2,
                       help='environment map parameter')
    args = parser.parse_args()
    main(args)