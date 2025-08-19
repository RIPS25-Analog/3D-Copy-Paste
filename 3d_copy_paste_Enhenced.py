"""
Multi-Object Scene Generation with 3D Placement and Collision Detection
Academic implementation for synthetic dataset creation with YOLO annotations
"""

import bpy
from mathutils import Vector, Matrix
from math import pi, radians
import numpy as np
import random
import json
import os
from PIL import Image
import cv2
import argparse
import time
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO


def suppress_blender_output() -> None:
    """Suppress Blender console output for clean execution."""
    try:
        import logging
        logging.getLogger('bpy').setLevel(logging.ERROR)
        bpy.app.debug_value = 0
        try:
            if hasattr(bpy.context, 'preferences') and hasattr(bpy.context.preferences, 'system'):
                system_prefs = bpy.context.preferences.system
                if hasattr(system_prefs, 'use_scripts_auto_execute'):
                    system_prefs.use_scripts_auto_execute = False
        except:
            pass
    except Exception:
        pass


def configure_rendering_engine() -> None:
    """Configure Blender rendering engine with optimal settings."""
    bpy.context.scene.render.engine = 'CYCLES'
    try:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'
    except:
        pass
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.cycles.preview_samples = 32
    bpy.context.scene.cycles.use_denoising = True


def apply_enhanced_materials(obj: bpy.types.Object, obj_dir: str, obj_name: str, object_index: int) -> None:
    """
    Apply enhanced materials with realistic textures and neutral fallback colors.
    
    Args:
        obj: Blender object to apply materials to
        obj_dir: Directory containing object files
        obj_name: Base name of the object
        object_index: Index for color selection
    """
    if not obj.data:
        return
        
    neutral_colors = [
        (0.7, 0.6, 0.5, 1.0),  # Beige
        (0.6, 0.6, 0.6, 1.0),  # Light Gray
        (0.5, 0.4, 0.3, 1.0),  # Brown
        (0.6, 0.5, 0.4, 1.0),  # Light Brown
        (0.4, 0.4, 0.4, 1.0),  # Dark Gray
        (0.7, 0.7, 0.6, 1.0),  # Off-white
        (0.5, 0.5, 0.6, 1.0),  # Blue-gray
        (0.6, 0.6, 0.5, 1.0),  # Olive
    ]
    
    try:
        if not hasattr(obj.data, 'materials') or len(obj.data.materials) == 0:
            mat = bpy.data.materials.new(name=f"{obj.name}_Material")
            obj.data.materials.append(mat)
    except Exception:
        return
    
    texture_extensions = ['.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tiff']
    
    for i, mat_slot in enumerate(obj.data.materials):
        try:
            if mat_slot is None:
                mat = bpy.data.materials.new(name=f"{obj.name}_Material_{i}")
                obj.data.materials[i] = mat
            else:
                mat = mat_slot
            
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            for node in nodes:
                nodes.remove(node)
            
            principled = nodes.new(type='ShaderNodeBsdfPrincipled')
            output = nodes.new(type='ShaderNodeOutputMaterial')
            links.new(principled.outputs['BSDF'], output.inputs['Surface'])
            
            texture_found = False
            for ext in texture_extensions:
                texture_patterns = [
                    f"{obj_name}{ext}", f"{obj_name}_diffuse{ext}", f"{obj_name}_albedo{ext}",
                    f"{obj_name}_color{ext}", f"{obj_name}_basecolor{ext}", f"{obj_name}_base_color{ext}",
                    f"material_0{ext}", f"material{ext}", f"texture{ext}", f"diffuse{ext}"
                ]
                
                for pattern in texture_patterns:
                    texture_path = os.path.join(obj_dir, pattern)
                    if os.path.exists(texture_path):
                        try:
                            if texture_path not in bpy.data.images:
                                image = bpy.data.images.load(texture_path)
                            else:
                                image = bpy.data.images[texture_path]
                            
                            if image.size[0] > 0 and image.size[1] > 0:
                                tex_node = nodes.new(type='ShaderNodeTexImage')
                                tex_node.image = image
                                tex_node.location = (-300, 0)
                                links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                                
                                if image.channels == 4:
                                    links.new(tex_node.outputs['Alpha'], principled.inputs['Alpha'])
                                    mat.blend_method = 'BLEND'
                                
                                texture_found = True
                                break
                        except Exception:
                            continue
                
                if texture_found:
                    break
            
            if not texture_found:
                color_idx = object_index % len(neutral_colors)
                chosen_color = neutral_colors[color_idx]
                principled.inputs['Base Color'].default_value = chosen_color
            
            principled.inputs['Alpha'].default_value = 1.0
            principled.inputs['Roughness'].default_value = 0.4
            principled.inputs['Metallic'].default_value = 0.0
            principled.inputs['Specular'].default_value = 0.5
            
            if texture_found and image.channels == 4:
                mat.blend_method = 'OPAQUE'
                mat.alpha_threshold = 0.1
            
        except Exception:
            continue


def load_and_insert_object(filepath: str, object_index: int) -> tuple:
    """
    Load 3D object from file with enhanced error handling.
    
    Args:
        filepath: Path to the .obj file
        object_index: Index for naming and material assignment
        
    Returns:
        Tuple of (imported_object, bounding_box_object) or (None, None) if failed
    """
    obj_dir = os.path.dirname(filepath)
    obj_name = os.path.splitext(os.path.basename(filepath))[0]
    
    initial_objects = set(bpy.data.objects.keys())
    
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            bpy.ops.import_scene.obj(
                filepath=filepath,
                use_edges=True,
                use_smooth_groups=True,
                use_split_objects=False,
                use_split_groups=False,
                use_groups_as_vgroups=False,
                use_image_search=True,
                split_mode='ON',
                global_clamp_size=0,
                axis_forward='-Z',
                axis_up='Y'
            )
        
        bpy.context.view_layer.update()
        
        current_objects = set(bpy.data.objects.keys())
        new_object_names = current_objects - initial_objects
        
        if not new_object_names:
            return None, None
        
        imported_objects = [bpy.data.objects[name] for name in new_object_names]
        mesh_objects = [obj for obj in imported_objects if obj.type == 'MESH' and obj.data is not None]
        
        if not mesh_objects:
            return None, None
        
        inobject = max(mesh_objects, key=lambda obj: len(obj.data.vertices) if obj.data else 0)
        
        if not inobject.data or len(inobject.data.vertices) == 0:
            return None, None
        
        unique_name = f"InsertedObject_{object_index}_{random.randint(1000,9999)}"
        inobject.name = unique_name
        
        bpy.context.view_layer.objects.active = inobject
        inobject.select_set(True)
        
        apply_enhanced_materials(inobject, obj_dir, obj_name, object_index)
        
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.mesh.primitive_cube_add()
        bbox = bpy.context.active_object
        
        if bbox is None:
            return None, None
        
        bbox.name = f"{unique_name}_bbox"
        bbox.location = inobject.location.copy()
        bbox.dimensions = inobject.dimensions.copy()
        bbox.hide_render = True
        
        if bbox.dimensions.length == 0:
            bbox.dimensions = Vector((0.1, 0.1, 0.1))
        
        inobject.select_set(False)
        bbox.select_set(False)
        bpy.context.view_layer.objects.active = None
        
        return inobject, bbox
        
    except Exception:
        current_objects = set(bpy.data.objects.keys())
        new_objects = current_objects - initial_objects
        for obj_name in new_objects:
            if obj_name in bpy.data.objects:
                try:
                    bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
                except:
                    pass
        return None, None


def calculate_object_bottom_z(obj: bpy.types.Object) -> float:
    """
    Calculate the actual world Z coordinate of the object's bottom after transformations.
    
    Args:
        obj: Blender object
        
    Returns:
        Z coordinate of the object's bottom
    """
    bpy.context.view_layer.update()
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return min(corner.z for corner in bbox_corners)


def calculate_object_top_z(obj: bpy.types.Object) -> float:
    """
    Calculate the actual world Z coordinate of the object's top.
    
    Args:
        obj: Blender object
        
    Returns:
        Z coordinate of the object's top
    """
    bpy.context.view_layer.update()
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return max(corner.z for corner in bbox_corners)


def calculate_object_bounds_3d(obj: bpy.types.Object) -> dict:
    """
    Calculate object's 3D bounding box in world coordinates.
    
    Args:
        obj: Blender object
        
    Returns:
        Dictionary containing bounding box coordinates and dimensions
    """
    bpy.context.view_layer.update()
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    min_x = min(corner.x for corner in bbox_corners)
    max_x = max(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    min_z = min(corner.z for corner in bbox_corners)
    max_z = max(corner.z for corner in bbox_corners)
    
    return {
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'min_z': min_z, 'max_z': max_z,
        'center_x': (min_x + max_x) / 2,
        'center_y': (min_y + min_y) / 2,
        'center_z': (min_z + max_z) / 2,
        'width': max_x - min_x,
        'length': max_y - min_y,
        'height': max_z - min_z
    }


def detect_3d_collision(new_obj_bounds: dict, existing_objects: list, min_distance: float = 0.5) -> dict:
    """
    Perform 3D collision detection between new object and existing objects.
    
    Args:
        new_obj_bounds: Bounding box dictionary for new object
        existing_objects: List of existing Blender objects
        min_distance: Minimum required distance between objects
        
    Returns:
        Dictionary containing collision detection results
    """
    collision_info = {
        'has_collision': False,
        'collision_objects': [],
        'min_distance_found': float('inf'),
        'collision_details': []
    }
    
    new_center = Vector((new_obj_bounds['center_x'], new_obj_bounds['center_y'], new_obj_bounds['center_z']))
    new_max_dim = max(new_obj_bounds['width'], new_obj_bounds['length'], new_obj_bounds['height'])
    
    for existing_obj in existing_objects:
        existing_bounds = calculate_object_bounds_3d(existing_obj)
        existing_center = Vector((existing_bounds['center_x'], existing_bounds['center_y'], existing_bounds['center_z']))
        existing_max_dim = max(existing_bounds['width'], existing_bounds['length'], existing_bounds['height'])
        
        distance_3d = (new_center - existing_center).length
        required_distance = (new_max_dim + existing_max_dim) / 2 + min_distance
        
        collision_info['min_distance_found'] = min(collision_info['min_distance_found'], distance_3d)
        
        if distance_3d < required_distance:
            collision_info['has_collision'] = True
            collision_info['collision_objects'].append(existing_obj)
            
            collision_detail = {
                'object': existing_obj.name,
                'distance': distance_3d,
                'required': required_distance,
                'overlap': required_distance - distance_3d
            }
            collision_info['collision_details'].append(collision_detail)
    
    return collision_info


def load_available_surfaces(scene_id: str, root_path: str) -> list:
    """
    Load all available surfaces for object placement with floor priority.
    
    Args:
        scene_id: Scene identifier
        root_path: Root directory path
        
    Returns:
        List of surface dictionaries sorted by priority
    """
    available_surfaces = []
    
    enhanced_plane_path = os.path.join(root_path, 'plane/{}/enhanced_plane_statistics.json'.format(scene_id))
    
    if os.path.exists(enhanced_plane_path):
        with open(enhanced_plane_path, 'r') as fh:
            plane_data = json.load(fh)
        
        if 'floor_id' in plane_data and plane_data['floor_id'] in plane_data['horizontal_planes']:
            floor_id = plane_data['floor_id']
            floor_info = {
                'type': 'floor',
                'stats': plane_data['horizontal_planes'][floor_id],
                'id': floor_id,
                'priority': 1
            }
            available_surfaces.append(floor_info)
        
        for surface_type in ['table', 'desk', 'counter', 'surface']:
            if surface_type in plane_data.get('surfaces', {}):
                for surface in plane_data['surfaces'][surface_type]:
                    surface_info = {
                        'type': surface_type,
                        'stats': surface['stats'],
                        'id': surface['id'],
                        'priority': 2
                    }
                    available_surfaces.append(surface_info)
    
    if not available_surfaces:
        floor_plane_path = os.path.join(root_path, 'plane/{}/floor_plane_statistics_noceiling_threshold0.04.json'.format(scene_id))
        if os.path.exists(floor_plane_path):
            with open(floor_plane_path, 'r') as fh:
                plane_data = json.load(fh)
            floor_info = {
                'type': 'floor',
                'stats': plane_data['floor'],
                'id': plane_data.get('floor_id', 'floor'),
                'priority': 1
            }
            available_surfaces.append(floor_info)
    
    return available_surfaces


def calculate_surface_height(surface_info: dict) -> float:
    """
    Calculate the correct surface height for object placement.
    
    Args:
        surface_info: Surface information dictionary
        
    Returns:
        Surface height in world coordinates
    """
    if surface_info['type'] == 'floor':
        return surface_info['stats']['mean'][2]
    else:
        return surface_info['stats']['max'][2]


def place_object_with_six_sided_rotation(inobject: bpy.types.Object, bbox: bpy.types.Object, 
                                        surface_info: dict, position_x: float, position_y: float, 
                                        scale: float, rotation_z: float, rotation_variety: float = 0.7) -> float:
    """
    Place object with six-sided rotation capability and perfect ground contact.
    
    Args:
        inobject: Blender object to place
        bbox: Bounding box object
        surface_info: Surface information dictionary
        position_x: X position coordinate
        position_y: Y position coordinate
        scale: Object scale factor
        rotation_z: Z-axis rotation angle
        rotation_variety: Probability of applying random rotation (0-1)
        
    Returns:
        Final bottom Z coordinate after placement
    """
    inobject.rotation_mode = 'XYZ'
    inobject.rotation_euler[2] = rotation_z
    
    if random.random() < rotation_variety:
        rotation_choice = random.choice([
            'bottom', 'top', 'front', 'back', 'left', 'right'
        ])
        
        rotation_map = {
            'bottom': (0, 0),
            'top': (pi, 0),
            'front': (pi/2, 0),
            'back': (-pi/2, 0),
            'left': (0, pi/2),
            'right': (0, -pi/2)
        }
        
        x_rot, y_rot = rotation_map[rotation_choice]
        inobject.rotation_euler[0] = x_rot
        inobject.rotation_euler[1] = y_rot
    else:
        inobject.rotation_euler[0] = 0
        inobject.rotation_euler[1] = 0
    
    inobject.location.x = position_x
    inobject.location.y = position_y
    
    surface_height = calculate_surface_height(surface_info)
    
    temp_z = surface_height + 2.0
    inobject.location.z = temp_z
    bpy.context.view_layer.update()
    
    rotated_bottom_z = calculate_object_bottom_z(inobject)
    bottom_offset_from_center = rotated_bottom_z - inobject.location.z
    
    target_center_z = surface_height - bottom_offset_from_center
    inobject.location.z = target_center_z
    
    bbox.location = inobject.location
    bbox.rotation_euler = inobject.rotation_euler
    
    bpy.context.view_layer.update()
    final_bottom_z = calculate_object_bottom_z(inobject)
    
    return final_bottom_z


def find_collision_free_placement(surface_info: dict, bbox_dims: Vector, all_placed_objects: list, 
                                 obj_idx: int, scale_range: tuple = (1.2, 2.8), max_attempts: int = 2000) -> dict:
    """
    Find collision-free placement for object on surface.
    
    Args:
        surface_info: Surface information dictionary
        bbox_dims: Bounding box dimensions
        all_placed_objects: List of already placed objects
        obj_idx: Object index for identification
        scale_range: Tuple of (min_scale, max_scale)
        max_attempts: Maximum placement attempts
        
    Returns:
        Dictionary containing placement parameters or failure indication
    """
    min_distance = 0.3
    
    for attempt in range(max_attempts):
        margin = 0.15
        test_x = random.uniform(
            surface_info['stats']['min'][0] + margin,
            surface_info['stats']['max'][0] - margin
        )
        test_y = random.uniform(
            surface_info['stats']['min'][1] + margin,
            surface_info['stats']['max'][1] - margin
        )
        
        test_scale = random.uniform(scale_range[0], scale_range[1])
        test_rotation_z = random.uniform(-pi, pi)
        
        surface_z = calculate_surface_height(surface_info)
        scaled_height = bbox_dims[2] * test_scale
        test_z = surface_z + scaled_height / 2
        
        scaled_width = bbox_dims[0] * test_scale
        scaled_length = bbox_dims[1] * test_scale
        
        temp_bounds = {
            'min_x': test_x - scaled_width/2,
            'max_x': test_x + scaled_width/2,
            'min_y': test_y - scaled_length/2,
            'max_y': test_y + scaled_length/2,
            'min_z': test_z - scaled_height/2,
            'max_z': test_z + scaled_height/2,
            'center_x': test_x,
            'center_y': test_y,
            'center_z': test_z,
            'width': scaled_width,
            'length': scaled_length,
            'height': scaled_height
        }
        
        collision_info = detect_3d_collision(temp_bounds, all_placed_objects, min_distance=min_distance)
        
        if not collision_info['has_collision']:
            return {
                'position_x': test_x,
                'position_y': test_y,
                'scale': test_scale,
                'rotation_z': test_rotation_z,
                'surface_id': surface_info['id'],
                'surface_type': surface_info['type'],
                'has_collision': False,
                'min_distance_achieved': collision_info['min_distance_found']
            }
    
    surface_center_x = surface_info['stats']['mean'][0]
    surface_center_y = surface_info['stats']['mean'][1]
    fallback_scale = scale_range[0] * 0.9
    
    return {
        'position_x': surface_center_x,
        'position_y': surface_center_y,
        'scale': fallback_scale,
        'rotation_z': 0,
        'surface_id': surface_info['id'],
        'surface_type': surface_info['type'],
        'has_collision': True,
        'fallback': True,
        'min_distance_achieved': 0
    }


def load_target_objects_configuration(json_path: str) -> dict:
    """
    Load target objects configuration from JSON file.
    
    Args:
        json_path: Path to JSON configuration file
        
    Returns:
        Dictionary containing object categories and their model IDs
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {
            'can': [],
            'toy_car': [],
            'distractor': []
        }


def create_object_selection_plan(target_objects_data: dict, num_objects_per_scene: int) -> list:
    """
    Create object selection plan with individual model selection.
    
    Args:
        target_objects_data: Dictionary containing available object models
        num_objects_per_scene: Target number of objects per scene
        
    Returns:
        List of selected object information dictionaries
    """
    if not (3 <= num_objects_per_scene <= 6):
        num_objects_per_scene = max(3, min(6, num_objects_per_scene))
    
    max_target_objects = min(2, num_objects_per_scene - 1)
    num_target_objects = random.randint(0, max_target_objects)
    num_distractors = num_objects_per_scene - num_target_objects
    
    selected_objects = []
    
    if num_target_objects > 0:
        target_pool = []
        for obj_id in target_objects_data['can']:
            target_pool.append({'id': obj_id, 'class': 'can'})
        for obj_id in target_objects_data['toy_car']:
            target_pool.append({'id': obj_id, 'class': 'toy_car'})
        
        if len(target_pool) >= num_target_objects:
            selected_targets = random.sample(target_pool, num_target_objects)
            selected_objects.extend(selected_targets)
        else:
            selected_objects.extend(target_pool)
            num_distractors += num_target_objects - len(target_pool)
    
    if num_distractors > 0:
        distractor_pool = [{'id': obj_id, 'class': 'distractor'} for obj_id in target_objects_data['distractor']]
        if len(distractor_pool) >= num_distractors:
            selected_distractors = random.sample(distractor_pool, num_distractors)
            selected_objects.extend(selected_distractors)
        else:
            selected_objects.extend(distractor_pool)
    
    random.shuffle(selected_objects)
    return selected_objects


def calculate_2d_bounding_box(obj: bpy.types.Object, cam: bpy.types.Object, scene: bpy.types.Scene) -> tuple:
    """
    Calculate 2D bounding box of object in screen coordinates.
    
    Args:
        obj: Blender object
        cam: Camera object
        scene: Blender scene
        
    Returns:
        Tuple of (min_x, min_y, max_x, max_y) or None if invalid
    """
    from bpy_extras.object_utils import world_to_camera_view
    
    bpy.context.view_layer.update()
    corners_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    screen_coords = []
    for corner in corners_world:
        co_2d = world_to_camera_view(scene, cam, corner)
        screen_x = co_2d.x * scene.render.resolution_x
        screen_y = (1 - co_2d.y) * scene.render.resolution_y
        screen_coords.append((screen_x, screen_y))
    
    if screen_coords:
        xs = [coord[0] for coord in screen_coords]
        ys = [coord[1] for coord in screen_coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = scene.render.resolution_x
        height = scene.render.resolution_y
        min_x = max(0, min(min_x, width))
        max_x = max(0, min(max_x, width))
        min_y = max(0, min(min_y, height))
        max_y = max(0, min(max_y, height))
        
        return min_x, min_y, max_x, max_y
    
    return None


def convert_to_yolo_format(bbox_coords: tuple, image_width: int, image_height: int, class_id: int = 0) -> tuple:
    """
    Convert bounding box coordinates to YOLO format.
    
    Args:
        bbox_coords: Tuple of (min_x, min_y, max_x, max_y)
        image_width: Image width in pixels
        image_height: Image height in pixels
        class_id: YOLO class identifier
        
    Returns:
        Tuple of (class_id, center_x_norm, center_y_norm, width_norm, height_norm) or None
    """
    if bbox_coords is None:
        return None
    
    min_x, min_y, max_x, max_y = bbox_coords
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    width = max_x - min_x
    height = max_y - min_y
    
    center_x_norm = center_x / image_width
    center_y_norm = center_y / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    
    return class_id, center_x_norm, center_y_norm, width_norm, height_norm


def create_class_mapping() -> dict:
    """
    Create mapping from class names to YOLO class IDs.
    
    Returns:
        Dictionary mapping class names to integer IDs
    """
    return {
        'can': 0, 
        'toy_car': 1, 
        'distractor': 2
    }


def validate_yolo_annotation(bbox_data: tuple, min_size: float = 0.01) -> bool:
    """
    Validate YOLO bounding box annotation.
    
    Args:
        bbox_data: Tuple of YOLO format data
        min_size: Minimum acceptable size threshold
        
    Returns:
        Boolean indicating validity
    """
    if bbox_data is None:
        return False
    
    class_id, center_x, center_y, width, height = bbox_data
    
    if not (0.05 <= center_x <= 0.95 and 0.05 <= center_y <= 0.95):
        return False
    
    if width < min_size or height < min_size:
        return False
    
    return True


def save_yolo_annotations(bbox_data_list: list, scene_id: str, iteration: int, output_dir: str) -> None:
    """
    Save YOLO format annotations to file.
    
    Args:
        bbox_data_list: List of YOLO format bounding box data
        scene_id: Scene identifier
        iteration: Iteration number
        output_dir: Output directory path
    """
    if not bbox_data_list:
        return
    
    annotation_lines = []
    for bbox_data in bbox_data_list:
        if bbox_data is not None:
            class_id, center_x, center_y, width, height = bbox_data
            annotation_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    if annotation_lines:
        annotation_file = os.path.join(output_dir, f"{scene_id}_{iteration}.txt")
        with open(annotation_file, 'w') as f:
            f.write('\n'.join(annotation_lines) + '\n')


def create_annotated_image(image_path: str, bbox_data_list: list, class_names: list, output_path: str) -> None:
    """
    Create annotated image with bounding boxes.
    
    Args:
        image_path: Path to input image
        bbox_data_list: List of YOLO format bounding box data
        class_names: List of class names corresponding to bounding boxes
        output_path: Path for output annotated image
    """
    if not bbox_data_list:
        return
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return
        
        height, width = img.shape[:2]
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
        
        for idx, (bbox_data, class_name) in enumerate(zip(bbox_data_list, class_names)):
            if bbox_data is None:
                continue
            
            class_id, center_x, center_y, bbox_width, bbox_height = bbox_data
            color = colors[idx % len(colors)]
            
            center_x_px = int(center_x * width)
            center_y_px = int(center_y * height) 
            bbox_width_px = int(bbox_width * width)
            bbox_height_px = int(bbox_height * height)
            
            x1 = int(center_x_px - bbox_width_px / 2)
            y1 = int(center_y_px - bbox_height_px / 2)
            x2 = int(center_x_px + bbox_width_px / 2)
            y2 = int(center_y_px + bbox_height_px / 2)
            
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            label = f"{class_name} ({idx+1})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            label_y = y1 - 10 if y1 > 30 else y2 + 25
            
            cv2.rectangle(img, (x1, label_y - label_size[1] - 5), 
                        (x1 + label_size[0] + 5, label_y + 5), color, -1)
            cv2.putText(img, label, (x1 + 2, label_y - 2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.circle(img, (center_x_px, center_y_px), 4, (255, 255, 255), -1)
        
        cv2.imwrite(output_path, img)
        
    except Exception:
        pass


def ensure_object_visibility(inobject: bpy.types.Object, bbox: bpy.types.Object, 
                           cam: bpy.types.Object, scene: bpy.types.Scene, 
                           min_size_norm: float = 0.18) -> float:
    """
    Ensure object is visible and appropriately sized in camera view.
    
    Args:
        inobject: Blender object
        bbox: Bounding box object
        cam: Camera object
        scene: Scene object
        min_size_norm: Minimum normalized size threshold
        
    Returns:
        Final scale factor applied
    """
    from bpy_extras.object_utils import world_to_camera_view
    
    bpy.context.view_layer.update()
    
    co_2d = world_to_camera_view(scene, cam, inobject.location)
    
    if not (0.1 <= co_2d.x <= 0.9 and 0.1 <= co_2d.y <= 0.9 and co_2d.z > 0):
        cam_location = cam.location
        cam_forward = cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        
        better_position = cam_location + cam_forward * 3.0
        inobject.location.x = better_position.x + random.uniform(-1, 1)
        inobject.location.y = better_position.y + random.uniform(-1, 1)
        bbox.location.x = inobject.location.x
        bbox.location.y = inobject.location.y
    
    bbox_corners = [inobject.matrix_world @ Vector(corner) for corner in inobject.bound_box]
    screen_coords = []
    
    for corner in bbox_corners:
        co_2d = world_to_camera_view(scene, cam, corner)
        screen_x = co_2d.x * scene.render.resolution_x
        screen_y = co_2d.y * scene.render.resolution_y
        screen_coords.append((screen_x, screen_y))
    
    if screen_coords:
        min_x = min(coord[0] for coord in screen_coords)
        max_x = max(coord[0] for coord in screen_coords)
        min_y = min(coord[1] for coord in screen_coords)
        max_y = max(coord[1] for coord in screen_coords)
        
        width_norm = (max_x - min_x) / scene.render.resolution_x
        height_norm = (max_y - min_y) / scene.render.resolution_y
        
        if width_norm < min_size_norm or height_norm < min_size_norm:
            scale_factor = min_size_norm / max(width_norm, height_norm)
            scale_factor = min(scale_factor, 3.2)
            
            new_scale = inobject.scale[0] * scale_factor
            inobject.scale = Vector((new_scale, new_scale, new_scale))
            bbox.scale = inobject.scale
            
            bpy.context.view_layer.update()
    
    return inobject.scale[0]


def initialize_blender_scene() -> None:
    """Initialize and configure Blender scene for object insertion."""
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except:
        pass
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    bpy.context.view_layer.update()


def create_shadow_plane(obj: bpy.types.Object, surface_info: dict) -> bpy.types.Object:
    """
    Create shadow plane for individual object.
    
    Args:
        obj: Blender object to create shadow for
        surface_info: Surface information dictionary
        
    Returns:
        Created shadow plane object
    """
    obj_bounds = calculate_object_bounds_3d(obj)
    
    plane_location = Vector((obj_bounds['center_x'], obj_bounds['center_y'], obj_bounds['min_z'] - 0.002))
    
    bpy.ops.mesh.primitive_plane_add(size=1, location=plane_location)
    shadow_plane = bpy.context.active_object
    shadow_plane.name = f"ShadowPlane_{obj.name}"
    
    shadow_size = max(obj_bounds['width'], obj_bounds['length']) * 1.2
    shadow_plane.scale = Vector((shadow_size, shadow_size, 1.0))
    
    shadow_plane.is_shadow_catcher = True
    shadow_plane.hide_viewport = True
    
    return shadow_plane


def configure_balanced_lighting(all_objects: list) -> None:
    """
    Configure balanced lighting for all objects in scene.
    
    Args:
        all_objects: List of all objects to illuminate
    """
    if not all_objects:
        return
    
    existing_lights = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']
    for light in existing_lights:
        try:
            bpy.data.objects.remove(light, do_unlink=True)
        except:
            pass
    
    center_pos = Vector((0, 0, 0))
    min_z = float('inf')
    max_z = float('-inf')
    
    for obj in all_objects:
        center_pos += obj.location
        obj_bounds = calculate_object_bounds_3d(obj)
        min_z = min(min_z, obj_bounds['min_z'])
        max_z = max(max_z, obj_bounds['max_z'])
    
    center_pos /= len(all_objects)
    
    # Key light
    bpy.ops.object.light_add(type='SUN', location=(center_pos.x, center_pos.y - 3, max_z + 3))
    key_light = bpy.context.active_object
    key_light.name = "KeyLight_Main"
    key_light.data.energy = 2.5
    key_light.rotation_euler = (radians(45), radians(15), radians(30))
    
    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(center_pos.x + 2, center_pos.y + 2, max_z + 1))
    fill_light = bpy.context.active_object
    fill_light.name = "FillLight_Area"
    fill_light.data.energy = 50.0
    fill_light.data.size = 1.5
    
    direction = center_pos - fill_light.location
    fill_light.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # Ambient light
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    ambient_light = bpy.context.active_object
    ambient_light.name = "AmbientLight_Gentle"
    ambient_light.data.energy = 0.8
    ambient_light.rotation_euler = (radians(15), radians(30), radians(45))


def configure_environment_lighting(scene_id: str, root_path: str, intensity_strength: int) -> None:
    """
    Configure environment lighting with HDR maps or default setup.
    
    Args:
        scene_id: Scene identifier
        root_path: Root directory path
        intensity_strength: Lighting intensity parameter
    """
    envmap_root = os.path.join(root_path, 'envmap')
    
    existing_envmaps = [
        os.path.join(envmap_root, f'{scene_id}_envmap0.png'),
        os.path.join(envmap_root, f'{scene_id}_envmap1.png'),
        os.path.join(envmap_root, f'{scene_id}.png')
    ]
    
    env_map_filepath = None
    for envmap_path in existing_envmaps:
        if os.path.exists(envmap_path):
            env_map_filepath = envmap_path
            break
    
    if env_map_filepath:
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                bpy.ops.image.open(filepath=env_map_filepath)
            
            env_map = bpy.data.images[os.path.basename(env_map_filepath)]
            
            world = bpy.context.scene.world
            if not world:
                world = bpy.data.worlds.new("Environment_World")
                bpy.context.scene.world = world
            
            world.use_nodes = True
            nodes = world.node_tree.nodes
            nodes.clear()
            
            output_node = nodes.new(type='ShaderNodeOutputWorld')
            background_node = nodes.new(type='ShaderNodeBackground')
            environment_texture_node = nodes.new(type='ShaderNodeTexEnvironment')
            mapping_node = nodes.new(type='ShaderNodeMapping')
            texture_coord_node = nodes.new(type='ShaderNodeTexCoord')
            
            environment_texture_node.image = env_map
            mapping_node.inputs[2].default_value[2] = radians(180)
            background_node.inputs[1].default_value = intensity_strength * 0.3
            
            links = world.node_tree.links
            links.new(texture_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
            links.new(mapping_node.outputs['Vector'], environment_texture_node.inputs['Vector'])
            links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])
            links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
            
        except Exception:
            setup_default_environment(intensity_strength)
    else:
        setup_default_environment(intensity_strength)


def setup_default_environment(strength: float = 1.0) -> None:
    """
    Set up default environment lighting.
    
    Args:
        strength: Lighting strength parameter
    """
    try:
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("Default_World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        node_tree = world.node_tree
        node_tree.nodes.clear()
        
        background_node = node_tree.nodes.new('ShaderNodeBackground')
        background_node.inputs['Color'].default_value = (0.1, 0.1, 0.12, 1.0)
        background_node.inputs['Strength'].default_value = strength * 0.25
        
        output_node = node_tree.nodes.new('ShaderNodeOutputWorld')
        node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
        
    except Exception:
        pass


def main(args: argparse.Namespace) -> None:
    """
    Main execution function for multi-object scene generation.
    
    Args:
        args: Command line arguments namespace
    """
    suppress_blender_output()
    configure_rendering_engine()
    
    root_path = args.root_path
    obj_root_path = args.obj_root_path
    
    with open(os.path.join(root_path, 'train_data_idx.txt')) as f:
        train_data_ids_raw = f.readlines()
    train_data_ids = ["{:06d}".format(int(id.split('\n')[0])) for id in train_data_ids_raw]
    
    target_objects_json_path = os.path.join(obj_root_path, 'objaverse.json')
    target_objects_data = load_target_objects_configuration(target_objects_json_path)
    
    max_iter = args.max_iter
    num_objects_per_scene = args.num_objects_per_scene
    random.seed(args.random_seed)
    istrength = args.istrength
    
    out_folder_name = "Pace_train_O_2"
    
    dirs = ['label', 'inserted_foreground', 'compositional_image', 'annotated_images', 'yolo_annotations', 'insert_object_log']
    for dir_name in dirs:
        os.makedirs(os.path.join(root_path, out_folder_name, dir_name), exist_ok=True)
    
    processing_times = []
    
    for iteration in range(max_iter):
        for index, scene_id in enumerate(train_data_ids):
            start_time = time.time()
            
            output_path = os.path.join(root_path, out_folder_name, 'compositional_image', f'{scene_id}_{iteration}.png')
            if os.path.exists(output_path):
                continue
            
            try:
                initialize_blender_scene()
                
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    bpy.ops.wm.open_mainfile(filepath=os.path.join(root_path, 'insertion_template.blend'))
                
                bpy.ops.object.select_all(action='SELECT')
                bpy.data.objects["Camera"].select_set(False)
                bpy.ops.object.delete()
                
                # Camera configuration
                with open(os.path.join(root_path, 'calib', f'{scene_id}.txt')) as f:
                    lines = f.readlines()
                
                R_raw = np.array([float(ele) for ele in lines[0].split()]).reshape((3, 3)).T
                flip_yz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                R = np.matmul(R_raw, flip_yz)
                matrix_world = np.eye(4)
                matrix_world[:3, :3] = R
                cam = bpy.data.objects['Camera']
                cam.matrix_world = Matrix(matrix_world)
                cam.location = Vector([0, 0, 0])
                
                K = np.array([float(ele) for ele in lines[1].split()]).reshape((3, 3)).T
                cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
                
                scene = bpy.context.scene
                scene.render.resolution_x = int(cx * 2)
                scene.render.resolution_y = int(cy * 2)
                bpy.data.cameras[bpy.data.cameras.keys()[0]].sensor_width = 36
                alpha = scene.render.resolution_x / bpy.data.cameras[bpy.data.cameras.keys()[0]].sensor_width
                bpy.data.cameras[bpy.data.cameras.keys()[0]].lens = fx / alpha
                
                all_surfaces = load_available_surfaces(scene_id, root_path)
                if not all_surfaces:
                    continue
                
                all_surfaces.sort(key=lambda s: s['priority'])
                
                all_objects = []
                all_bboxes = []
                all_class_names = []
                all_yolo_data = []
                all_shadow_planes = []
                
                successful_placements = 0
                
                selected_object_info_list = create_object_selection_plan(target_objects_data, num_objects_per_scene)
                
                for obj_idx, obj_info in enumerate(selected_object_info_list):
                    placed = False
                    max_attempts_per_object = 15
                    
                    for attempt in range(max_attempts_per_object):
                        try:
                            obj_id = obj_info['id']
                            insert_class = obj_info['class']
                            obj_path = os.path.join(obj_root_path, obj_id, obj_id + '.obj')
                            
                            inobject, bbox = load_and_insert_object(obj_path, obj_idx)
                            if inobject is None:
                                continue
                            
                            placement_result = None
                            for surface in all_surfaces:
                                placement_result = find_collision_free_placement(
                                    surface, bbox.dimensions, all_objects, obj_idx, 
                                    scale_range=(1.2, 2.8)
                                )
                                
                                if placement_result and not placement_result.get('has_collision', True):
                                    break
                            
                            if placement_result is None:
                                bpy.data.objects.remove(inobject, do_unlink=True)
                                bpy.data.objects.remove(bbox, do_unlink=True)
                                continue
                            
                            surface_for_placement = None
                            for surface in all_surfaces:
                                if surface['id'] == placement_result['surface_id']:
                                    surface_for_placement = surface
                                    break
                            
                            if surface_for_placement is None:
                                bpy.data.objects.remove(inobject, do_unlink=True)
                                bpy.data.objects.remove(bbox, do_unlink=True)
                                continue
                            
                            final_scale = placement_result['scale']
                            inobject.scale = Vector((final_scale, final_scale, final_scale))
                            bbox.scale = inobject.scale
                            
                            final_z = place_object_with_six_sided_rotation(
                                inobject, bbox, surface_for_placement,
                                placement_result['position_x'],
                                placement_result['position_y'],
                                final_scale,
                                placement_result['rotation_z'],
                                rotation_variety=0.8
                            )
                            
                            final_scale_after_visibility = ensure_object_visibility(inobject, bbox, cam, scene, min_size_norm=0.16)
                            
                            bbox_coords = calculate_2d_bounding_box(inobject, cam, scene)
                            if bbox_coords:
                                class_mapping = create_class_mapping()
                                class_id = class_mapping.get(insert_class, 2)
                                yolo_bbox = convert_to_yolo_format(
                                    bbox_coords, scene.render.resolution_x, 
                                    scene.render.resolution_y, class_id
                                )
                                
                                if validate_yolo_annotation(yolo_bbox):
                                    all_objects.append(inobject)
                                    all_bboxes.append(bbox)
                                    all_class_names.append(insert_class)
                                    all_yolo_data.append(yolo_bbox)
                                    
                                    shadow_plane = create_shadow_plane(inobject, surface_for_placement)
                                    all_shadow_planes.append(shadow_plane)
                                    
                                    successful_placements += 1
                                    placed = True
                                    break
                                else:
                                    bpy.data.objects.remove(inobject, do_unlink=True)
                                    bpy.data.objects.remove(bbox, do_unlink=True)
                            else:
                                bpy.data.objects.remove(inobject, do_unlink=True)
                                bpy.data.objects.remove(bbox, do_unlink=True)
                                
                        except Exception:
                            continue
                    
                    if not placed:
                        print(f"Warning: Could not place object {obj_idx + 1} ({obj_info['id']}) after {max_attempts_per_object} attempts")
                
                if successful_placements == 0:
                    continue
                
                configure_balanced_lighting(all_objects)
                configure_environment_lighting(scene_id, root_path, istrength)
                
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    bpy.context.scene.render.filepath = os.path.join(
                        root_path, out_folder_name, 'inserted_foreground', f'{scene_id}_{iteration}.png'
                    )
                    bpy.ops.render.render(write_still=True)
                
                # Composite with background
                foreground_path = os.path.join(root_path, out_folder_name, 'inserted_foreground', f'{scene_id}_{iteration}.png')
                background_path = os.path.join(root_path, 'image', f'{scene_id}.jpg')
                composite_path = os.path.join(root_path, out_folder_name, 'compositional_image', f'{scene_id}_{iteration}.png')
                
                frontImage = Image.open(foreground_path).convert("RGBA")
                background = Image.open(background_path)
                background.paste(frontImage, (0, 0), frontImage)
                background.save(composite_path, format="png")
                
                # Save annotations
                if all_yolo_data:
                    yolo_dir = os.path.join(root_path, out_folder_name, 'yolo_annotations')
                    save_yolo_annotations(all_yolo_data, scene_id, iteration, yolo_dir)
                    
                    annotated_path = os.path.join(root_path, out_folder_name, 'annotated_images', f'{scene_id}_{iteration}.png')
                    create_annotated_image(composite_path, all_yolo_data, all_class_names, annotated_path)
                    
                    # Detailed logs
                    log_data = {
                        'scene_id': scene_id,
                        'iteration': iteration,
                        'num_objects_requested': len(selected_object_info_list),
                        'num_objects_placed': successful_placements,
                        'individual_model_selection': True,
                        'objects': []
                    }
                    
                    for i, (obj, cls, obj_info) in enumerate(zip(all_objects, all_class_names, selected_object_info_list[:successful_placements])):
                        obj_detail = {
                            'index': i,
                            'object_id': obj_info['id'],
                            'class': cls,
                            'position': [float(obj.location.x), float(obj.location.y), float(obj.location.z)],
                            'rotation_euler': [float(obj.rotation_euler.x), float(obj.rotation_euler.y), float(obj.rotation_euler.z)],
                            'rotation_degrees': [float(obj.rotation_euler.x*180/pi), float(obj.rotation_euler.y*180/pi), float(obj.rotation_euler.z*180/pi)],
                            'scale': float(obj.scale[0]),
                            'ground_contact_z': float(calculate_object_bottom_z(obj))
                        }
                        log_data['objects'].append(obj_detail)
                    
                    with open(os.path.join(root_path, out_folder_name, 'insert_object_log', f'{scene_id}_{iteration}.json'), 'w') as f:
                        json.dump(log_data, f, indent=4)
                    
                    with open(os.path.join(root_path, out_folder_name, 'label', f'{scene_id}_{iteration}.json'), 'w') as f:
                        json.dump(log_data, f, indent=4)
                
                # Cleanup
                bpy.ops.object.select_all(action='SELECT')
                bpy.data.objects["Camera"].select_set(False)
                bpy.ops.object.delete()
                
                end_time = time.time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)
                
                print(f"Scene {scene_id} completed: {successful_placements} objects placed in {processing_time:.2f}s")
                
            except Exception as e:
                print(f"Error processing scene {scene_id}: {e}")
    
    if processing_times:
        total_scenes = len(processing_times)
        total_time = sum(processing_times)
        avg_time = total_time / total_scenes
        print(f"Processing complete: {total_scenes} scenes in {total_time:.2f}s (avg: {avg_time:.2f}s per scene)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Object Scene Generation with Collision Detection')
    parser.add_argument('--root_path', 
                       default="/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/mmdetection3d/data/sunrgbd/sunrgbd_trainval",
                       help='Root path for SUN RGB-D data')
    parser.add_argument('--obj_root_path', 
                       default="/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/Pace/train_ready/obj",
                       help='Root path for 3D model data')
    parser.add_argument('--max_iter', type=int, default=1,
                       help='Number of iterations for each scene')
    parser.add_argument('--num_objects_per_scene', type=int, default=4,
                       help='Number of objects to place per scene')
    parser.add_argument('--random_seed', type=int, default=5142714,
                       help='Random seed for reproducibility')
    parser.add_argument('--ilog', type=int, default=2,
                       help='Environment map parameter')
    parser.add_argument('--istrength', type=int, default=2,
                       help='Environment map parameter')
    
    args = parser.parse_args()
    main(args)