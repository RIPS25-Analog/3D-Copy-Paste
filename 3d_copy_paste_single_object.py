# 🔧 MULTI-OBJECT FIXED EDITION - Multiple Objects + No Collisions + Better Randomization! 🔧
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

# ===== 🔧 NO FLOATING HELPER FUNCTIONS 🔧 =====

def get_object_actual_bottom_z(obj):
    """Get the actual world Z coordinate of the object's bottom after all transformations"""
    bpy.context.view_layer.update()
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return min(corner.z for corner in bbox_corners)

def create_shadow_plane_touching_object(inobject, surface_info, shadow_plane_size=10):
    """Create shadow plane that touches the object's bottom exactly - NO FLOATING!"""
    object_bottom_z = get_object_actual_bottom_z(inobject)
    
    plane_location = inobject.location.copy()
    plane_location[2] = object_bottom_z - 0.001  # Just 1mm below object bottom
    
    print(f"🔧 SHADOW FIX: Object bottom: {object_bottom_z:.6f}, Shadow: {plane_location[2]:.6f}, Gap: {(object_bottom_z - plane_location[2])*1000:.1f}mm")
    
    bpy.ops.mesh.primitive_plane_add(size=shadow_plane_size, enter_editmode=False, align='WORLD', location=plane_location)
    plane = bpy.context.active_object
    plane.name = "ShadowPlane"
    plane.is_shadow_catcher = True
    
    # Scale shadow plane based on object size
    obj_max_dim = max(inobject.dimensions.x, inobject.dimensions.y) * inobject.scale[0]
    if shadow_plane_size < obj_max_dim * 2:
        plane.scale = Vector((obj_max_dim * 0.5, obj_max_dim * 0.5, 1.0))
    
    return plane

# ===== 🐱 BALANCED + VISIBILITY PROTECTION FUNCTIONS 🐱 =====

def ensure_90_percent_visibility(inobject, bbox, surface_info, cam, scene, min_visibility=0.9):
    """
    🐱 VISIBILITY PROTECTION: Ensure at least 90% of object is visible (not hidden in cabinets/underground)
    """
    print(f"🐱 VISIBILITY CHECK: Ensuring {min_visibility:.0%} of object is visible...")
    
    bpy.context.view_layer.update()
    
    # Step 1: Check if object is underground (most common issue)
    surface_height = get_proper_surface_height(surface_info)
    obj_bottom_z = get_object_actual_bottom_z(inobject)
    obj_top_z = get_object_top_world_z(inobject)
    obj_height = obj_top_z - obj_bottom_z
    
    # Calculate how much is underground
    if obj_bottom_z < surface_height:
        underground_depth = surface_height - obj_bottom_z
        underground_ratio = underground_depth / obj_height
        
        if underground_ratio > 0.1:  # More than 10% underground
            print(f"⚠️  Object {underground_ratio:.1%} underground! Lifting to surface...")
            
            # Lift object so only 5% is below surface (realistic placement)
            target_underground = 0.05
            target_bottom_z = surface_height - (obj_height * target_underground)
            z_adjustment = target_bottom_z - obj_bottom_z
            
            inobject.location.z += z_adjustment
            bbox.location.z += z_adjustment
            bpy.context.view_layer.update()
            
            print(f"   Lifted object by {z_adjustment:.3f}m")
    
    # Step 2: Check camera visibility by sampling many points on object
    visibility_ratio = calculate_detailed_visibility_ratio(inobject, cam, scene)
    
    print(f"   Object visibility: {visibility_ratio:.1%}")
    
    if visibility_ratio < min_visibility:
        print(f"⚠️  Object only {visibility_ratio:.1%} visible! Moving to better position...")
        
        # Try different positioning strategies
        best_position = find_best_visible_position(inobject, bbox, surface_info, cam, scene, min_visibility)
        
        if best_position:
            inobject.location.x = best_position[0]
            inobject.location.y = best_position[1]
            inobject.location.z = best_position[2]
            bbox.location = inobject.location
            bpy.context.view_layer.update()
            
            # Re-check visibility
            new_visibility = calculate_detailed_visibility_ratio(inobject, cam, scene)
            print(f"   New visibility: {new_visibility:.1%}")
            
            if new_visibility >= min_visibility:
                print(f"🐱 ✅ VISIBILITY FIXED: Object now {new_visibility:.1%} visible!")
                return True
            else:
                print(f"🐱 ⚠️  Best effort: {new_visibility:.1%} visible (target was {min_visibility:.1%})")
                return new_visibility >= 0.7  # Accept if at least 70%
        else:
            print(f"🐱 ⚠️  Could not find better position, keeping current placement")
            return visibility_ratio >= 0.7  # Accept if at least 70%
    else:
        print(f"🐱 ✅ Object visibility is good!")
        return True

def get_object_top_world_z(obj):
    """Get the world Z coordinate of the object's top"""
    bpy.context.view_layer.update()
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return max(corner.z for corner in bbox_corners)

def calculate_detailed_visibility_ratio(obj, cam, scene):
    """
    Calculate what percentage of the object is visible by sampling many points
    """
    from bpy_extras.object_utils import world_to_camera_view
    
    bpy.context.view_layer.update()
    
    # Get object's 8 bounding box corners
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Add more sample points for better accuracy
    sample_points = []
    
    # Add all 8 corner points
    sample_points.extend(bbox_corners)
    
    # Add center points of each face
    min_corner = Vector([min(c[0] for c in bbox_corners), min(c[1] for c in bbox_corners), min(c[2] for c in bbox_corners)])
    max_corner = Vector([max(c[0] for c in bbox_corners), max(c[1] for c in bbox_corners), max(c[2] for c in bbox_corners)])
    center = (min_corner + max_corner) / 2
    
    # Face centers
    sample_points.extend([
        Vector([min_corner.x, center.y, center.z]),  # Left face
        Vector([max_corner.x, center.y, center.z]),  # Right face  
        Vector([center.x, min_corner.y, center.z]),  # Back face
        Vector([center.x, max_corner.y, center.z]),  # Front face
        Vector([center.x, center.y, min_corner.z]),  # Bottom face
        Vector([center.x, center.y, max_corner.z]),  # Top face
        center  # Object center
    ])
    
    # Count visible points
    visible_count = 0
    total_count = len(sample_points)
    
    for point in sample_points:
        co_2d = world_to_camera_view(scene, cam, point)
        
        # Check if point is in camera view
        if (0.02 <= co_2d.x <= 0.98 and 0.02 <= co_2d.y <= 0.98 and co_2d.z > 0):
            visible_count += 1
    
    return visible_count / total_count if total_count > 0 else 0

def find_best_visible_position(inobject, bbox, surface_info, cam, scene, min_visibility, max_attempts=20):
    """
    Try to find a position where the object is at least min_visibility visible
    """
    surface_center_x = surface_info['stats']['mean'][0]
    surface_center_y = surface_info['stats']['mean'][1]
    surface_width = surface_info['stats']['max'][0] - surface_info['stats']['min'][0]
    surface_length = surface_info['stats']['max'][1] - surface_info['stats']['min'][1]
    
    best_position = None
    best_visibility = 0
    
    original_pos = inobject.location.copy()
    
    for attempt in range(max_attempts):
        # Try positions in expanding circles around surface center
        angle = (attempt / max_attempts) * 2 * pi
        radius = (attempt / max_attempts) * min(surface_width, surface_length) * 0.3
        
        test_x = surface_center_x + cos(angle) * radius
        test_y = surface_center_y + sin(angle) * radius
        
        # Ensure within surface bounds
        test_x = np.clip(test_x, 
                        surface_info['stats']['min'][0] + surface_width * 0.1,
                        surface_info['stats']['max'][0] - surface_width * 0.1)
        test_y = np.clip(test_y,
                        surface_info['stats']['min'][1] + surface_length * 0.1,
                        surface_info['stats']['max'][1] - surface_length * 0.1)
        
        # Calculate proper Z position
        surface_height = get_proper_surface_height(surface_info)
        obj_height = get_object_top_world_z(inobject) - get_object_actual_bottom_z(inobject)
        test_z = surface_height + obj_height * 0.1  # 10% above surface
        
        # Test this position
        inobject.location.x = test_x
        inobject.location.y = test_y
        inobject.location.z = test_z
        bpy.context.view_layer.update()
        
        visibility = calculate_detailed_visibility_ratio(inobject, cam, scene)
        
        if visibility > best_visibility:
            best_visibility = visibility
            best_position = (test_x, test_y, test_z)
            
            if visibility >= min_visibility:
                print(f"   Found good position at attempt {attempt + 1}: {visibility:.1%} visible")
                break
    
    # Restore original position temporarily
    inobject.location = original_pos
    bpy.context.view_layer.update()
    
    if best_position and best_visibility > calculate_detailed_visibility_ratio(inobject, cam, scene):
        return best_position
    else:
        return None

# ===== 🐱 BALANCED CAT PROTECTION SCALING FUNCTIONS 🐱 =====
def calculate_adaptive_object_scale_BALANCED(surface_info, bbox_dimensions, insertion_size_statistics,
                                  insert_class, min_coverage=0.15, max_coverage=0.5):
    """
    🐱 BALANCED VERSION: Reasonable minimum sizes (cat safety) with controlled maximum sizes
    """
    surface_area = surface_info['stats']['area']
    surface_width = surface_info['stats']['max'][0] - surface_info['stats']['min'][0]
    surface_length = surface_info['stats']['max'][1] - surface_info['stats']['min'][1]
    
    # Get reference height from statistics - BALANCED approach
    if insert_class in insertion_size_statistics:
        size_z = random.gauss(
            mu=insertion_size_statistics[insert_class][0],
            sigma=insertion_size_statistics[insert_class][1]
        )
        # 🐱 BALANCED: Reasonable size range - not too small, not too big
        size_z = np.clip(size_z, 0.12, 0.8)  # 12cm to 80cm height
    else:
        size_z = 0.3  # Default 30cm height
    
    reference_object_height = size_z * 2
    obj_shrink_factor = bbox_dimensions[2] / reference_object_height
    
    # Object base dimensions after initial scaling
    scaled_width = bbox_dimensions[0] / obj_shrink_factor
    scaled_length = bbox_dimensions[1] / obj_shrink_factor
    scaled_base_area = scaled_width * scaled_length
    
    # 🐱 BALANCED: Reasonable coverage allowance
    min_coverage = max(min_coverage, 0.2)   # At least 20% coverage
    max_coverage = min(max_coverage, 0.6)   # Max 60% coverage (reduced from 80%)
    
    # Calculate scale factors based on coverage
    min_scale = np.sqrt(surface_area * min_coverage / scaled_base_area)
    max_scale = np.sqrt(surface_area * max_coverage / scaled_base_area)
    
    # 🐱 BALANCED: Reasonable dimension constraints  
    width_scale = surface_width * 0.45 / scaled_width   # Up to 45% of surface width
    length_scale = surface_length * 0.45 / scaled_length # Up to 45% of surface length
    
    # Final scale is the minimum of all constraints
    max_allowed_scale = min(max_scale, width_scale, length_scale)
    
    # 🐱 BALANCED: Reasonable scale range
    if max_allowed_scale > min_scale:
        additional_scale = random.uniform(min_scale, max_allowed_scale)
    else:
        additional_scale = max_allowed_scale * 0.9
    
    # 🐱 BALANCED: Reasonable minimum and maximum bounds
    additional_scale = np.clip(additional_scale, 0.8, 3.0)  # Between 80% and 300% (much more reasonable!)
    
    print(f"🐱 BALANCED SCALE DEBUG - Class: {insert_class}")
    print(f"  Surface area: {surface_area:.2f}, dims: {surface_width:.2f} x {surface_length:.2f}")
    print(f"  Reference height: {size_z:.3f}, obj_shrink: {obj_shrink_factor:.3f}")
    print(f"  Scale range: {min_scale:.3f} - {max_allowed_scale:.3f}")
    print(f"  Final additional_scale: {additional_scale:.3f}")
    
    return obj_shrink_factor, additional_scale

def get_proper_surface_height(surface_info):
    """
    Get the correct surface height for object placement
    """
    if surface_info['type'] == 'floor':
        # For floors, use mean Z (not max, as floors can have noise)
        return surface_info['stats']['mean'][2]
    else:
        # For tables/surfaces, use max Z (top of the surface)
        return surface_info['stats']['max'][2]

# ===== 🔧 NO FLOATING OBJECT PLACEMENT FUNCTIONS 🔧 =====

def place_object_on_surface_WITH_RANDOM_ORIENTATION_NO_FLOAT(inobject, bbox, surface_info, best_parameter, final_scale, orientation_variety=0.7):
    """🎲 🔧 Random orientation placement with ZERO floating - perfect ground contact"""
    print(f"🎲 🔧 RANDOM ORIENTATION + NO FLOATING (variety: {orientation_variety:.1%})...")
    
    # Apply rotations first
    inobject.rotation_mode = 'XYZ'
    inobject.rotation_euler[2] = best_parameter['sample_rotation'] / 180 * pi
    
    # Apply random orientations
    if random.random() < orientation_variety:
        rotation_options = [0, 90, -90, 180]
        
        if random.random() < 0.6:  # 60% chance of X rotation
            random_x = random.choice(rotation_options) * (pi / 180)
            inobject.rotation_euler[0] = random_x
            print(f"   Applied X rotation: {random_x * 180 / pi:.0f}°")
        
        if random.random() < 0.6:  # 60% chance of Y rotation  
            random_y = random.choice(rotation_options) * (pi / 180)
            inobject.rotation_euler[1] = random_y
            print(f"   Applied Y rotation: {random_y * 180 / pi:.0f}°")
    
    # Position object
    inobject.location.x = best_parameter['sample_position_X']
    inobject.location.y = best_parameter['sample_position_Y']
    
    # Get surface height
    surface_height = get_proper_surface_height(surface_info)
    
    # CRITICAL: Calculate bottom position after ALL rotations
    temp_z = surface_height + 2.0
    inobject.location.z = temp_z
    bpy.context.view_layer.update()
    
    # Measure where the bottom actually is after rotation
    rotated_bottom_z = get_object_actual_bottom_z(inobject)
    bottom_offset_from_center = rotated_bottom_z - inobject.location.z
    
    # Position object so rotated bottom touches surface EXACTLY
    target_center_z = surface_height - bottom_offset_from_center
    inobject.location.z = target_center_z
    
    # Update bbox
    bbox.location = inobject.location
    bbox.rotation_euler = inobject.rotation_euler
    
    # Final verification
    bpy.context.view_layer.update()
    final_bottom_z = get_object_actual_bottom_z(inobject)
    surface_contact_gap = final_bottom_z - surface_height
    
    print(f"🔧 NO-FLOAT COMPLETE: Bottom at {final_bottom_z:.6f}, Surface at {surface_height:.6f}, Gap: {surface_contact_gap*1000:.1f}mm")
    
    return final_bottom_z

def BALANCED_collision_check_WITH_TRACKING(surface_info, bbox, insertion_size_statistics, insert_class, 
                                       all_objects_info_dict, obj_shrink_factor, additional_scale, obj_idx):
    """
    🆕 MULTI-OBJECT FIX: Collision check that includes ALL objects (original + previously inserted)
    """
    best_parameter = {}
    
    # Increase attempts for multi-object scenarios
    max_attempts = 2000  # Increased from 1200
    
    for i in range(max_attempts):
        # Better randomization per object
        random_offset = random.random()  # Add more randomness
        
        # 🐱 BALANCED: More reasonable scaling with variation
        if surface_info['type'] == 'floor':
            scale_multiplier = random.uniform(1.4 + random_offset * 0.3, 2.2 + random_offset * 0.5)
        elif surface_info['type'] in ['table', 'desk', 'counter', 'surface']:
            scale_multiplier = random.uniform(1.1 + random_offset * 0.2, 1.8 + random_offset * 0.4)
        else:
            scale_multiplier = random.uniform(1.2 + random_offset * 0.2, 2.0 + random_offset * 0.4)
        
        sample_scale = additional_scale * scale_multiplier
        
        # Better position sampling with more variation
        surface_std_X = surface_info['stats']['std'][0]
        surface_std_Y = surface_info['stats']['std'][1]
        surface_center_X = surface_info['stats']['mean'][0]
        surface_center_Y = surface_info['stats']['mean'][1]
        
        margin_x = (surface_info['stats']['max'][0] - surface_info['stats']['min'][0]) * 0.1
        margin_y = (surface_info['stats']['max'][1] - surface_info['stats']['min'][1]) * 0.1
        
        # Use different sampling strategies to increase diversity
        if i < max_attempts // 3:
            # First third: sample near center
            sample_position_X = random.gauss(surface_center_X, surface_std_X * 0.3)
            sample_position_Y = random.gauss(surface_center_Y, surface_std_Y * 0.3)
        elif i < 2 * max_attempts // 3:
            # Second third: sample more spread out
            sample_position_X = random.gauss(surface_center_X, surface_std_X * 0.6)
            sample_position_Y = random.gauss(surface_center_Y, surface_std_Y * 0.6)
        else:
            # Last third: sample uniformly across surface
            sample_position_X = random.uniform(surface_info['stats']['min'][0] + margin_x,
                                              surface_info['stats']['max'][0] - margin_x)
            sample_position_Y = random.uniform(surface_info['stats']['min'][1] + margin_y,
                                              surface_info['stats']['max'][1] - margin_y)
        
        # Ensure within bounds
        sample_position_X = np.clip(sample_position_X,
                                  surface_info['stats']['min'][0] + margin_x,
                                  surface_info['stats']['max'][0] - margin_x)
        sample_position_Y = np.clip(sample_position_Y,
                                  surface_info['stats']['min'][1] + margin_y,
                                  surface_info['stats']['max'][1] - margin_y)
        
        sample_rotation = random.uniform(-180, 180)
        
        # Random orientation parameters
        sample_rotation_x = 0
        sample_rotation_y = 0
        
        if random.random() < 0.7:
            if random.random() < 0.6:
                sample_rotation_x = random.choice([0, 90, -90, 180])
            if random.random() < 0.6:
                sample_rotation_y = random.choice([0, 90, -90, 180])
        
        if i == 0:
            best_parameter['iou'] = 1
            best_parameter['sample_scale'] = sample_scale
            best_parameter['sample_position_X'] = sample_position_X
            best_parameter['sample_position_Y'] = sample_position_Y
            best_parameter['sample_rotation'] = sample_rotation
            best_parameter['sample_rotation_x'] = sample_rotation_x
            best_parameter['sample_rotation_y'] = sample_rotation_y
        
        # Collision calculation
        sample_obj_shrink_factor = obj_shrink_factor * sample_scale
        max_dim = max(bbox.dimensions[0], bbox.dimensions[1], bbox.dimensions[2])
        sample_bbox_size = [max_dim / sample_obj_shrink_factor / 2,
                          max_dim / sample_obj_shrink_factor / 2]
        
        sample_bbox_info = [sample_position_X, sample_position_Y,
                          sample_bbox_size[0], sample_bbox_size[1], sample_rotation]
        
        # 🆕 KEY FIX: Check collision against ALL objects (original + inserted)
        collision_info = check_collision(all_objects_info_dict, sample_bbox_info)
        
        iou = 0.0
        for obj in collision_info.keys():
            if obj != 'bbox_corners':
                iou += collision_info[obj]['iou']
        
        # Accept position if no collision
        if iou <= 0.01:
            best_parameter['iou'] = iou
            best_parameter['sample_scale'] = sample_scale
            best_parameter['sample_position_X'] = sample_position_X
            best_parameter['sample_position_Y'] = sample_position_Y
            best_parameter['sample_rotation'] = sample_rotation
            best_parameter['sample_rotation_x'] = sample_rotation_x
            best_parameter['sample_rotation_y'] = sample_rotation_y
            print(f"   Found collision-free position for object {obj_idx} at attempt {i + 1}")
            break
        elif iou < best_parameter['iou']:
            best_parameter['iou'] = iou
            best_parameter['sample_scale'] = sample_scale
            best_parameter['sample_position_X'] = sample_position_X
            best_parameter['sample_position_Y'] = sample_position_Y
            best_parameter['sample_rotation'] = sample_rotation
            best_parameter['sample_rotation_x'] = sample_rotation_x
            best_parameter['sample_rotation_y'] = sample_rotation_y
    
    if best_parameter['iou'] > 0.01:
        print(f"   ⚠️  Could not find collision-free position, using best with IOU={best_parameter['iou']:.4f}")
    
    return best_parameter

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

def ensure_minimum_object_size(inobject, bbox, cam, scene, min_width_norm=0.140691, min_height_norm=0.057055):
    """
    🐱 CAT SAFETY: Ensure object meets minimum 2D bounding box size requirements
    """
    print(f"🐱 CAT SAFETY CHECK: Ensuring minimum object size...")
    print(f"   Required minimum: {min_width_norm:.1%} width x {min_height_norm:.1%} height")
    
    bpy.context.view_layer.update()
    
    # Get current 2D bounding box
    bbox_coords = get_2d_bounding_box(inobject, cam, scene)
    
    if bbox_coords is None:
        print("❌ Could not get 2D bounding box, applying emergency scale")
        emergency_scale = 2.5  # Reasonable emergency scale
        inobject.scale = Vector((emergency_scale, emergency_scale, emergency_scale))
        bbox.scale = inobject.scale
        return emergency_scale
    
    min_x, min_y, max_x, max_y = bbox_coords
    
    # Calculate current normalized dimensions
    image_width = scene.render.resolution_x
    image_height = scene.render.resolution_y
    
    current_width = max_x - min_x
    current_height = max_y - min_y
    
    current_width_norm = current_width / image_width
    current_height_norm = current_height / image_height
    
    print(f"   Current size: {current_width_norm:.1%} width x {current_height_norm:.1%} height")
    
    # Calculate required scale factors
    width_scale_factor = min_width_norm / current_width_norm if current_width_norm > 0 else 2.0
    height_scale_factor = min_height_norm / current_height_norm if current_height_norm > 0 else 2.0
    
    # Use the larger scale factor to ensure both dimensions meet minimum
    required_scale_factor = max(width_scale_factor, height_scale_factor, 1.0)  # Never scale down
    
    # 🐱 BALANCED: Cap the scale increase to prevent giant objects
    required_scale_factor = min(required_scale_factor, 3.0)  # Maximum 3x scale increase
    
    if required_scale_factor > 1.0:
        print(f"⚠️  Object TOO SMALL! Scaling up by {required_scale_factor:.2f}x")
        
        # Get current scale and apply additional scaling
        current_scale = inobject.scale[0]  # Assuming uniform scaling
        new_scale = current_scale * required_scale_factor
        
        # 🐱 BALANCED: Final safety check - don't let objects get too big
        if new_scale > 8.0:  # Maximum reasonable scale
            print(f"⚠️  Scale would be too large ({new_scale:.2f}), capping at 8.0")
            new_scale = 8.0
        
        # Apply the new scale
        inobject.scale = Vector((new_scale, new_scale, new_scale))
        bbox.scale = inobject.scale
        
        # Update and verify
        bpy.context.view_layer.update()
        
        # Check the result
        new_bbox_coords = get_2d_bounding_box(inobject, cam, scene)
        if new_bbox_coords:
            new_min_x, new_min_y, new_max_x, new_max_y = new_bbox_coords
            new_width_norm = (new_max_x - new_min_x) / image_width
            new_height_norm = (new_max_y - new_min_y) / image_height
            
            print(f"✅ BALANCED: New size: {new_width_norm:.1%} width x {new_height_norm:.1%} height")
            
            # Double check it meets requirements
            if new_width_norm >= min_width_norm and new_height_norm >= min_height_norm:
                print(f"🐱 CAT SAFE: Object now meets minimum size requirements!")
            else:
                print(f"⚠️  Still might be small, but applied maximum reasonable scale")
        
        return new_scale
    else:
        print(f"✅ Object already meets minimum size requirements")
        return inobject.scale[0]

def ensure_maximum_object_size(inobject, bbox, cam, scene, max_width_norm=0.5, max_height_norm=0.5):
    """
    🐱 HALF-IMAGE CONSTRAINT: Ensure object doesn't cover more than 50% of the image
    """
    print(f"🐱 HALF-IMAGE CHECK: Ensuring maximum object size...")
    print(f"   Maximum allowed: {max_width_norm:.1%} width x {max_height_norm:.1%} height")
    
    bpy.context.view_layer.update()
    
    # Get current 2D bounding box with retry mechanism
    bbox_coords = get_2d_bounding_box(inobject, cam, scene)
    
    # If bounding box calculation fails, apply emergency constraint
    if bbox_coords is None:
        print("⚠️  Could not get 2D bounding box, applying emergency size constraint")
        current_scale = inobject.scale[0]
        # Apply conservative scaling to ensure it's not too big
        emergency_scale = min(current_scale, 2.0)  # Cap at 2x scale
        inobject.scale = Vector((emergency_scale, emergency_scale, emergency_scale))
        bbox.scale = inobject.scale
        print(f"   Applied emergency scale: {emergency_scale:.3f}")
        return emergency_scale
    
    min_x, min_y, max_x, max_y = bbox_coords
    
    # Calculate current normalized dimensions
    image_width = scene.render.resolution_x
    image_height = scene.render.resolution_y
    
    current_width = max_x - min_x
    current_height = max_y - min_y
    
    current_width_norm = current_width / image_width
    current_height_norm = current_height / image_height
    
    print(f"   Current size: {current_width_norm:.1%} width x {current_height_norm:.1%} height")
    
    # Check if object is too large (more strict enforcement)
    if current_width_norm > max_width_norm or current_height_norm > max_height_norm:
        # Calculate scale reduction needed
        width_scale_factor = max_width_norm / current_width_norm if current_width_norm > max_width_norm else 1.0
        height_scale_factor = max_height_norm / current_height_norm if current_height_norm > max_height_norm else 1.0
        
        # Use the smaller scale factor to ensure both dimensions are within limits
        required_scale_factor = min(width_scale_factor, height_scale_factor)
        
        # Add additional safety margin (scale down a bit more)
        required_scale_factor *= 0.9  # 10% additional margin for safety
        
        print(f"⚠️  Object TOO LARGE! Scaling down by {1/required_scale_factor:.2f}x")
        
        # Get current scale and apply reduction
        current_scale = inobject.scale[0]  # Assuming uniform scaling
        new_scale = current_scale * required_scale_factor
        
        # Apply the new scale
        inobject.scale = Vector((new_scale, new_scale, new_scale))
        bbox.scale = inobject.scale
        
        # Update and verify with multiple attempts
        bpy.context.view_layer.update()
        
        # Verification with retry
        for attempt in range(3):
            new_bbox_coords = get_2d_bounding_box(inobject, cam, scene)
            if new_bbox_coords:
                new_min_x, new_min_y, new_max_x, new_max_y = new_bbox_coords
                new_width_norm = (new_max_x - new_min_x) / image_width
                new_height_norm = (new_max_y - new_min_y) / image_height
                
                print(f"✅ HALF-IMAGE CONSTRAINT: New size: {new_width_norm:.1%} width x {new_height_norm:.1%} height")
                
                # Double check - if still too big, scale down more aggressively
                if new_width_norm > max_width_norm or new_height_norm > max_height_norm:
                    print(f"⚠️  Still too large, applying more aggressive scaling...")
                    additional_scale_factor = min(max_width_norm / new_width_norm, max_height_norm / new_height_norm) * 0.8
                    new_scale *= additional_scale_factor
                    inobject.scale = Vector((new_scale, new_scale, new_scale))
                    bbox.scale = inobject.scale
                    bpy.context.view_layer.update()
                else:
                    print(f"🐱 HALF-IMAGE SAFE: Object now within 50% image limits!")
                    break
                break
            else:
                # If verification fails, assume success but log it
                print(f"   Verification attempt {attempt + 1} failed, but scaling was applied")
                if attempt == 2:
                    print(f"🐱 APPLIED SCALING: Cannot verify but constraint was enforced")
        
        return new_scale
    else:
        print(f"✅ Object size is within 50% image limits")
        return inobject.scale[0]

def ENHANCED_object_placement_WITH_RANDOM_ORIENTATION_NO_FLOAT(inobject, bbox, surface_info, best_parameter, cam, scene):
    """🎲 🔧 Enhanced placement with random orientations and NO FLOATING"""
    
    # Step 1: Calculate final scale (same as before)
    base_scale = best_parameter['sample_scale']
    
    if surface_info['type'] == 'floor':
        min_scale = 1.0
        max_scale = 2.5
    elif surface_info['type'] in ['table', 'desk', 'counter', 'surface']:
        min_scale = 0.8
        max_scale = 2.0
    else:
        min_scale = 0.9
        max_scale = 2.2
    
    final_scale = np.clip(base_scale, min_scale, max_scale)
    
    # Apply scale
    inobject.scale = Vector((final_scale, final_scale, final_scale))
    bbox.scale = inobject.scale
    
    # Step 2: Apply random orientations and NO-FLOAT positioning
    actual_bottom_z = place_object_on_surface_WITH_RANDOM_ORIENTATION_NO_FLOAT(
        inobject, bbox, surface_info, best_parameter, final_scale, orientation_variety=0.7
    )
    
    # Step 3: All the existing protections (size, visibility, etc.)
    final_scale_after_minimum = ensure_minimum_object_size(
        inobject, bbox, cam, scene, 
        min_width_norm=0.140691,
        min_height_norm=0.057055
    )
    
    final_scale_after_maximum = ensure_maximum_object_size(
        inobject, bbox, cam, scene,
        max_width_norm=0.5,
        max_height_norm=0.5
    )
    
    visibility_success = ensure_90_percent_visibility(inobject, bbox, surface_info, cam, scene, min_visibility=0.9)
    
    # Step 4: Re-verify positioning after scaling changes WITH NO-FLOAT GUARANTEE
    if final_scale_after_maximum != final_scale:
        print(f"🔧 Re-positioning after scale change with NO-FLOAT guarantee...")
        current_rotations = inobject.rotation_euler.copy()
        place_object_on_surface_WITH_RANDOM_ORIENTATION_NO_FLOAT(
            inobject, bbox, surface_info, best_parameter, final_scale_after_maximum, orientation_variety=0.0
        )
        inobject.rotation_euler = current_rotations
    
    # Step 5: Final camera view adjustment WITH NO-FLOAT MAINTENANCE
    bpy.context.view_layer.update()
    from bpy_extras.object_utils import world_to_camera_view
    center_2d = world_to_camera_view(scene, cam, inobject.location)
    
    if not (0.1 <= center_2d.x <= 0.9 and 0.1 <= center_2d.y <= 0.9 and center_2d.z > 0):
        print(f"🔧 Final camera adjustment with ground contact maintenance...")
        surface_center_x = surface_info['stats']['mean'][0]
        surface_center_y = surface_info['stats']['mean'][1]
        
        inobject.location.x += (surface_center_x - inobject.location.x) * 0.2
        inobject.location.y += (surface_center_y - inobject.location.y) * 0.2
        bbox.location.x = inobject.location.x
        bbox.location.y = inobject.location.y
        
        # CRITICAL: Maintain ground contact after adjustment
        surface_height = get_proper_surface_height(surface_info)
        temp_z = surface_height + 1.0
        inobject.location.z = temp_z
        bpy.context.view_layer.update()
        
        current_bottom_z = get_object_actual_bottom_z(inobject)
        bottom_offset = current_bottom_z - temp_z
        inobject.location.z = surface_height - bottom_offset
        bbox.location.z = inobject.location.z
        
        # Verify final ground contact
        bpy.context.view_layer.update()
        final_bottom_check = get_object_actual_bottom_z(inobject)
        final_gap = final_bottom_check - surface_height
        print(f"🔧 Final adjustment complete: Gap = {final_gap*1000:.1f}mm")
    
    return final_scale_after_maximum

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

def save_yolo_annotation_multi(bbox_data_list, scene_id, iter, output_dir):
    """🆕 Save YOLO format annotation for MULTIPLE objects"""
    if not bbox_data_list:
        return
    
    # Create annotation string for all objects
    annotation_lines = []
    for bbox_data in bbox_data_list:
        if bbox_data is not None:
            class_id, center_x, center_y, width, height = bbox_data
            annotation_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    # Save to file
    if annotation_lines:
        annotation_file = os.path.join(output_dir, f"{scene_id}_{iter}.txt")
        with open(annotation_file, 'w') as f:
            f.write('\n'.join(annotation_lines) + '\n')

def draw_bounding_boxes_on_image_multi(image_path, bbox_data_list, class_names, output_path):
    """🆕 Draw MULTIPLE 2D bounding box annotations on the image"""
    if not bbox_data_list:
        return
    
    try:
        import cv2
        import numpy as np
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return
        
        height, width = img.shape[:2]
        
        # Colors for different objects
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
        
        for idx, (bbox_data, class_name) in enumerate(zip(bbox_data_list, class_names)):
            if bbox_data is None:
                continue
            
            class_id, center_x, center_y, bbox_width, bbox_height = bbox_data
            
            # Get color for this object
            color = colors[idx % len(colors)]
            
            # Convert normalized YOLO coordinates to pixel coordinates
            center_x_px = int(center_x * width)
            center_y_px = int(center_y * height) 
            bbox_width_px = int(bbox_width * width)
            bbox_height_px = int(bbox_height * height)
            
            # Calculate corners
            x1 = int(center_x_px - bbox_width_px / 2)
            y1 = int(center_y_px - bbox_height_px / 2)
            x2 = int(center_x_px + bbox_width_px / 2)
            y2 = int(center_y_px + bbox_height_px / 2)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            # Draw bounding box rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{class_name} ({idx+1})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Position label
            label_y = y1 - 10 if y1 > 30 else y2 + 25
            label_x = x1
            
            # Draw label background
            cv2.rectangle(img, (label_x, label_y - label_size[1] - 5), 
                        (label_x + label_size[0] + 5, label_y + 5), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (label_x + 2, label_y - 2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw center point
            cv2.circle(img, (center_x_px, center_y_px), 4, (255, 255, 255), -1)
        
        # Save annotated image
        cv2.imwrite(output_path, img)
        
    except Exception as e:
        pass

def validate_yolo_bbox(bbox_data, min_size=0.01):
    """Validate YOLO bounding box to ensure it's visible and reasonable"""
    if bbox_data is None:
        return False
    
    class_id, center_x, center_y, width, height = bbox_data
    
    # Check if center is within image bounds (more lenient margins)
    if not (0.05 <= center_x <= 0.95 and 0.05 <= center_y <= 0.95):
        return False
    
    # Check if size is reasonable
    if width < min_size or height < min_size:
        return False
    
    return True

def cleanup_existing_invalid_files(root_path, out_folder_name):
    """
    🐱 CLEANUP UTILITY: Remove existing files that don't have corresponding annotated images
    """
    print(f"🐱 CHECKING EXISTING FILES: Looking for invalid files to cleanup...")
    
    compositional_dir = os.path.join(root_path, out_folder_name, 'compositional_image')
    annotated_dir = os.path.join(root_path, out_folder_name, 'annotated_images')
    
    if not os.path.exists(compositional_dir):
        print("   No compositional images found to check")
        return
    
    # Get all compositional images
    compositional_files = [f for f in os.listdir(compositional_dir) if f.endswith('.png')]
    
    invalid_files = []
    for comp_file in compositional_files:
        # Check if corresponding annotated image exists
        annotated_file = os.path.join(annotated_dir, comp_file)
        if not os.path.exists(annotated_file):
            invalid_files.append(comp_file)
    
    if invalid_files:
        print(f"   Found {len(invalid_files)} invalid files to cleanup")
        
        deleted_total = 0
        for invalid_file in invalid_files:
            scene_iter = invalid_file.replace('.png', '')
            
            # List of files to delete for this scene_iter
            files_to_delete = [
                os.path.join(root_path, out_folder_name, 'compositional_image', f'{scene_iter}.png'),
                os.path.join(root_path, out_folder_name, 'inserted_foreground', f'{scene_iter}.png'),
                os.path.join(root_path, out_folder_name, 'label', f'{scene_iter}.json'),
                os.path.join(root_path, out_folder_name, 'insert_object_log', f'{scene_iter}.json'),
                os.path.join(root_path, out_folder_name, 'yolo_annotations', f'{scene_iter}.txt')
            ]
            
            # Delete files if they exist
            for file_path in files_to_delete:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_total += 1
                except Exception as e:
                    print(f"     Could not delete {file_path}: {e}")
        
        print(f"   ✅ Cleaned up {deleted_total} invalid files")
    else:
        print(f"   ✅ All existing files are valid - no cleanup needed")

def main(args):
    '''Prepare'''
    # Initialize timing
    all_times = []
    
    # Suppress Blender output
    suppress_blender_output()
    
    # Setup better material rendering
    setup_better_material_rendering()
    
    root_path = args.root_path
    # Load all train scene data
    with open(os.path.join(root_path, 'train_data_idx.txt')) as f:
        train_data_ids_raw = f.readlines()
    train_data_ids = ["{:06d}".format(int(id.split('\n')[0])) for id in train_data_ids_raw]
    
    # Load objaverse pool
    with open('/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/Pace/obj_resized/test/objaverse.json') as f:
        objaverse_objects_dict = json.load(f)
    objaverse_object_name_list = list(objaverse_objects_dict.keys())
    obj_root_path = args.obj_root_path
    
    # Insertion hyperparameters
    max_iter = args.max_iter
    base_random_seed = args.random_seed  # 🆕 Store base seed
    ilog = args.ilog
    istrength = args.istrength
    insertion_mode = args.insertion_mode
    surface_selection_mode = args.surface_selection_mode
    min_coverage = args.min_coverage
    max_coverage = args.max_coverage
    
    out_folder_name = "aaaaaaaaaaaaaidk"
    
    # Create output directories
    os.makedirs(os.path.join(root_path, out_folder_name, 'label'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'inserted_foreground'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'compositional_image'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'annotated_images'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'envmap'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'insert_object_log'), exist_ok=True)
    os.makedirs(os.path.join(root_path, out_folder_name, 'yolo_annotations'), exist_ok=True)
    
    # Cleanup invalid files
    cleanup_existing_invalid_files(root_path, out_folder_name)
    
    envmap_root = os.path.join(root_path, 'envmap')
    
    '''Go over each scene'''
    for index, scene_id in enumerate(train_data_ids):
        for iter in range(1):  # Single iteration per scene, but with multiple objects
            # 🆕 MULTI-OBJECT: Create unique seed for this scene-iteration combo
            scene_seed = base_random_seed + index * 1000 + iter * 100
            random.seed(scene_seed)
            print(f"\n🆕 MULTI-OBJECT SCENE {scene_id}, iter {iter} (seed: {scene_seed})")
            
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
                # Open the blend file
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
                
                '''2. 🆕 MULTI-OBJECT Insertion operation'''
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
                
                # 🆕 MULTI-OBJECT: Track all objects (original + inserted)
                all_objects_info_dict = ori_GT_object_info_dict.copy()
                
                # 🆕 MULTI-OBJECT: Lists to store data for all objects
                inserted_objects = []  # Store all inserted Blender objects
                inserted_bboxes = []   # Store all bounding boxes
                inserted_classes = []  # Store class names
                yolo_bbox_list = []    # Store YOLO annotations
                insert_logs = []       # Store insertion logs
                
                # 🆕 MULTI-OBJECT: Determine number of objects to insert
                num_objects_to_insert = max_iter  # Use max_iter as number of objects per scene
                print(f"   Attempting to insert {num_objects_to_insert} objects...")
                
                # 🆕 MULTI-OBJECT: Insert multiple objects
                for obj_idx in range(num_objects_to_insert):
                    print(f"\n   === Inserting object {obj_idx + 1}/{num_objects_to_insert} ===")
                    
                    # 🆕 Add more randomization per object
                    object_seed = scene_seed + obj_idx * 10
                    random.seed(object_seed)
                    
                    # Select object class
                    insert_class = select_inserted_object_class(objaverse_object_name_list, ori_cate_list, insertion_mode)
                    
                    # Select and load object
                    def select_inserted_object(class_name):
                        candidates_dict = objaverse_objects_dict[class_name]
                        obj_name = random.choice(candidates_dict)
                        obj_path = os.path.join(obj_root_path, obj_name, obj_name + '.obj')
                        return obj_path
                    
                    inserted_object_path = select_inserted_object(insert_class)
                    insert_log_dict = {}
                    insert_log_dict['object_index'] = obj_idx
                    insert_log_dict['insert_class'] = insert_class
                    insert_log_dict['inserted_object_path'] = inserted_object_path
                    insert_log_dict['surface_type'] = surface_info['type']
                    insert_log_dict['surface_id'] = surface_info['id']
                    insert_log_dict['surface_area'] = surface_info['stats']['area']
                    
                    # Load object
                    try:
                        inobject, bbox = load_insert_object(filepath=inserted_object_path)
                        
                        # Give unique names to avoid conflicts
                        inobject.name = f"InsertedObject_{obj_idx}"
                        bbox.name = f"BBox_{obj_idx}"
                        
                    except Exception as e:
                        print(f"   Failed to load object {inserted_object_path}: {e}")
                        continue
                    
                    # Calculate adaptive scale
                    obj_shrink_factor, additional_scale = calculate_adaptive_object_scale_BALANCED(
                        surface_info, bbox.dimensions, insertion_size_statistics,
                        insert_class, min_coverage, max_coverage
                    )
                    insert_log_dict['obj_shrink_factor'] = obj_shrink_factor
                    insert_log_dict['adaptive_scale'] = additional_scale
                    
                    # 🆕 MULTI-OBJECT: Collision check against ALL objects
                    best_parameter = BALANCED_collision_check_WITH_TRACKING(
                        surface_info, bbox, insertion_size_statistics, insert_class,
                        all_objects_info_dict, obj_shrink_factor, additional_scale, obj_idx
                    )
                    
                    insert_log_dict['best_parameter'] = best_parameter
                    insert_log_dict['random_orientation_applied'] = True
                    insert_log_dict['rotation_x'] = best_parameter.get('sample_rotation_x', 0)
                    insert_log_dict['rotation_y'] = best_parameter.get('sample_rotation_y', 0)
                    insert_log_dict['rotation_z'] = best_parameter.get('sample_rotation', 0)
                    
                    # Place object
                    scene = bpy.context.scene
                    cam = bpy.data.objects['Camera']
                    
                    final_scale = ENHANCED_object_placement_WITH_RANDOM_ORIENTATION_NO_FLOAT(
                        inobject, bbox, surface_info, best_parameter, cam, scene
                    )
                    
                    # 🆕 MULTI-OBJECT: Add this object to tracking dictionary
                    object_key = f"inserted_{obj_idx}_{insert_class}"
                    all_objects_info_dict[object_key] = [
                        0, 0, 0, 0,  # First 4 values not used
                        bbox.location[0],  # center_X
                        bbox.location[1],  # center_Y
                        bbox.location[2],  # center_Z
                        bbox.dimensions[1] * final_scale / 2,  # width
                        bbox.dimensions[0] * final_scale / 2,  # length
                        bbox.dimensions[2] * final_scale / 2,  # height
                        np.cos(best_parameter['sample_rotation'] / 180 * pi),  # for angle calculation
                        np.sin(best_parameter['sample_rotation'] / 180 * pi)   # for angle calculation
                    ]
                    
                    # Get 2D bounding box
                    bbox_coords = get_2d_bounding_box(inobject, cam, scene)
                    if bbox_coords:
                        class_mapping = create_class_mapping()
                        class_id = class_mapping.get(insert_class, 0)
                        
                        image_width = scene.render.resolution_x
                        image_height = scene.render.resolution_y
                        yolo_bbox = convert_to_yolo_format(bbox_coords, image_width, image_height, class_id)
                        
                        if validate_yolo_bbox(yolo_bbox, min_size=0.001):
                            # Store data for this object
                            inserted_objects.append(inobject)
                            inserted_bboxes.append(bbox)
                            inserted_classes.append(insert_class)
                            yolo_bbox_list.append(yolo_bbox)
                            insert_logs.append(insert_log_dict)
                            
                            print(f"   ✅ Object {obj_idx + 1} placed successfully!")
                        else:
                            print(f"   ⚠️  Object {obj_idx + 1} invalid bbox, removing...")
                            # Remove object if invalid
                            bpy.data.objects.remove(inobject, do_unlink=True)
                            bpy.data.objects.remove(bbox, do_unlink=True)
                    else:
                        print(f"   ⚠️  Object {obj_idx + 1} no bbox detected, removing...")
                        # Remove object if no bbox
                        bpy.data.objects.remove(inobject, do_unlink=True)
                        bpy.data.objects.remove(bbox, do_unlink=True)
                
                # 🆕 Check if we have at least one valid object
                if not inserted_objects:
                    print(f"   ❌ No valid objects were inserted for scene {scene_id}")
                    continue
                
                print(f"\n   🎯 Successfully inserted {len(inserted_objects)}/{num_objects_to_insert} objects!")
                
                # Save combined insertion log
                combined_log = {
                    'scene_id': scene_id,
                    'iteration': iter,
                    'num_objects_inserted': len(inserted_objects),
                    'objects': insert_logs
                }
                with open(os.path.join(root_path, out_folder_name, 'insert_object_log', '{}_{}.json'.format(scene_id, iter)), "w") as outfile:
                    json.dump(combined_log, outfile, indent=4)
                
                # Save 3D info for all objects
                all_3d_info = {
                    'num_objects': len(inserted_objects),
                    'objects': []
                }
                
                for idx, (obj, bbox_obj, cls) in enumerate(zip(inserted_objects, inserted_bboxes, inserted_classes)):
                    obj_3d_info = {
                        'object_index': idx,
                        'class': cls,
                        'centroid_X': bbox_obj.location[0],
                        'centroid_Y': bbox_obj.location[1],
                        'centroid_Z': bbox_obj.location[2],
                        'width': bbox_obj.dimensions[1] * obj.scale[0] / 2,
                        'length': bbox_obj.dimensions[0] * obj.scale[0] / 2,
                        'height': bbox_obj.dimensions[2] * obj.scale[0] / 2,
                        'rotation_euler': [float(obj.rotation_euler.x), 
                                         float(obj.rotation_euler.y), 
                                         float(obj.rotation_euler.z)],
                        'scale': float(obj.scale[0]),
                        'surface_type': surface_info['type']
                    }
                    all_3d_info['objects'].append(obj_3d_info)
                
                with open(os.path.join(root_path, out_folder_name, 'label', '{}_{}.json'.format(scene_id, iter)), "w") as outfile:
                    json.dump(all_3d_info, outfile, indent=4)
                
                # Add lighting
                if inserted_objects:
                    # Use center of all objects for lighting
                    avg_location = Vector([0, 0, 0])
                    for obj in inserted_objects:
                        avg_location += obj.location
                    avg_location /= len(inserted_objects)
                    add_debug_lighting(avg_location)
                
                # Create shadow planes for all objects
                for obj in inserted_objects:
                    shadow_plane = create_shadow_plane_touching_object(obj, surface_info, shadow_plane_size=10)
                
                '''3. Dynamic illumination'''
                backimg = mpimg.imread(os.path.join(root_path, 'image', '{}.jpg'.format(scene_id)))
                
                # Use first object's position for environment map selection
                if inserted_objects:
                    first_object = inserted_objects[0]
                    object_center_world = Vector((first_object.location[0], first_object.location[1], surface_info['stats']['mean'][2]))
                    screen_x, screen_y = world_to_screen_coords(object_center_world, cam)
                    object_center_2d_img = Vector((screen_x, backimg.shape[0] - screen_y))
                    object_center_2d_img[0] = np.clip(object_center_2d_img[0], 0, backimg.shape[1])
                    object_center_2d_img[1] = np.clip(object_center_2d_img[1], 0, backimg.shape[0])
                    
                    # Load environment map
                    envCol = 160
                    envRow = 120
                    nh = backimg.shape[0]
                    nw = backimg.shape[1]
                    if nh < nw:
                        newW = envCol
                        newH = int(float(newW) / float(nw) * nh)
                    zoom_ratio = float(nw) / float(newW)
                    envmap_index = Vector((object_center_2d_img[1] / zoom_ratio, object_center_2d_img[0] / zoom_ratio))
                    
                    # Try to find existing environment maps
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
                
                # 🆕 Save MULTI-OBJECT YOLO annotations
                if yolo_bbox_list:
                    yolo_annotations_dir = os.path.join(root_path, out_folder_name, 'yolo_annotations')
                    save_yolo_annotation_multi(yolo_bbox_list, scene_id, iter, yolo_annotations_dir)
                
                # 🆕 Create MULTI-OBJECT annotated version
                annotated_image_created = False
                if yolo_bbox_list:
                    annotated_path = os.path.join(root_path, out_folder_name, 'annotated_images', '{}_{}.png'.format(scene_id, iter))
                    draw_bounding_boxes_on_image_multi(composite_path, yolo_bbox_list, inserted_classes, annotated_path)
                    annotated_image_created = True
                
                # Cleanup if no valid objects
                if not annotated_image_created:
                    print(f"⚠️  No valid objects detected for scene {scene_id}_{iter}, cleaning up files...")
                    
                    files_to_delete = [
                        os.path.join(root_path, out_folder_name, 'compositional_image', '{}_{}.png'.format(scene_id, iter)),
                        os.path.join(root_path, out_folder_name, 'inserted_foreground', '{}_{}.png'.format(scene_id, iter)),
                        os.path.join(root_path, out_folder_name, 'label', '{}_{}.json'.format(scene_id, iter)),
                        os.path.join(root_path, out_folder_name, 'insert_object_log', '{}_{}.json'.format(scene_id, iter)),
                        os.path.join(root_path, out_folder_name, 'yolo_annotations', '{}_{}.txt'.format(scene_id, iter))
                    ]
                    
                    deleted_count = 0
                    for file_path in files_to_delete:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                deleted_count += 1
                        except Exception as e:
                            print(f"   Could not delete {file_path}: {e}")
                    
                    print(f"   Cleaned up {deleted_count} files for failed insertion")
                else:
                    print(f"✅ {len(inserted_objects)} objects successfully inserted and annotated for scene {scene_id}_{iter}")
                
                # Clean up Blender scene
                bpy.ops.object.select_all(action='SELECT')
                bpy.data.objects["Camera"].select_set(False)
                bpy.ops.object.delete()
                
                # END TIMING
                end_time = time.time()
                processing_time = end_time - start_time
                all_times.append(processing_time)
                
                # Calculate and display average
                current_avg = sum(all_times) / len(all_times)
                
                if annotated_image_created:
                    print(f"🎯 MULTI-OBJECT SUCCESS: Scene {scene_id} with {len(inserted_objects)} objects ({processing_time:.2f}s) | Avg: {current_avg:.2f}s")
                else:
                    print(f"⚠️  CLEANUP COMPLETE: Removed invalid files for scene {scene_id} ({processing_time:.2f}s) | Avg: {current_avg:.2f}s")
                
            except Exception as e:
                print(f"❌ Error processing scene {scene_id}: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='🔧 MULTI-OBJECT EDITION - Multiple Objects + No Collisions + Better Randomization! 🔧')
    parser.add_argument('--root_path', default="/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/mmdetection3d/data/sunrgbd/sunrgbd_trainval",
                       help='root path for SUN RGB-D data')
    parser.add_argument('--obj_root_path', default="/home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/Pace/obj_resized/test",
                       help='root path for PACE data')
    parser.add_argument('--insertion_mode', default="context",
                       help='random: randomly insert any objects, context: only insert existing category objects')
    parser.add_argument('--surface_selection_mode', default='random',
                       choices=['random', 'weighted', 'floor_only', 'tables_only'],
                       help='How to select placement surface')
    parser.add_argument('--max_iter', type=int, default=4,
                       help='number of objects to insert per scene (MULTI-OBJECT MODE)')
    parser.add_argument('--random_seed', type=int, default=411521,
                       help='base random seed for reproduce')
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