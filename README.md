# 3D Copy-Paste

## Overview

This repository provides a pipeline for 3D Copy-Paste data augmentation by inserting realistic 3D objects into the SUN RGB-D dataset. The method generates synthetic training data for 2D object detection by compositing 3D models into real indoor scenes with physically plausible placement and lighting.

## Prerequisites

### Repository Dependencies

Please fork the following repositories under our workspace:

- **[mmdetection3d](https://github.com/rips25analog2025/mmdetection3d)** - 3D object detection framework
- **[InverseRenderingOfIndoorScene](https://github.com/rips25analog2025/InverseRenderingOfIndoorScene)** - Indoor scene inverse rendering
- **[RGBDPlaneDetection](https://github.com/rips25analog2025/RGBDPlaneDetection)** - Plane detection for object placement

## Environment Setup
### Software Used: 
python and MATLAB is require for this project

### Packages
please install the following packages
```bash
pip install numpy scipy matplotlib pillow

pip install torch torchvision
pip install tensorflow keras
pip install jax jaxlib

pip install opencv-python
pip install scikit-learn pandas h5py

pip install jupyter ipython
pip install flake8

pip install trimesh  # For advanced 3D mesh operations
pip install open3d   # Alternative 3D processing library

```

## Quick Start

> **Note:** For detailed explanations and insturctions of modifying the original 3D Copy-Paste implementation, please refer to [original README.md](Original_3D_Copy-Paste/Original_README.md).

### 1. Download SUN RGB-D Dataset
Go to [SUNRGBD dataset](https://rgbd.cs.princeton.edu/data/) and download:
SUNRGBD.zip;
SUNRGBDMeta2DBB_v2.mat;
SUNRGBDMeta3DBB_v2.mat 

### 2. Prepare 3D Models

Ensure your 3D models are in the supported format. 
```
3d_models/
    |   ├── class_name_1
    |   |   ├── instance1
    |   |   |   ├── instance1.obj
    |   |   |   ├── instance1.mtl
    |   |   |   └── instance1.png
    |   |   ├── instance2 
    |   |   └── ...   
    |   ├── class_name_2
    |   ├── class_name_3
    |   └── ...
```

### 3. Pre-Data Generation Preparation

```bash
bash Preparation.sh
```
This step orgnaized the naccessary imformation from the background images and the 3d models

### 4. Generate Synthetic Data

```bash
python 3d_copy_paste.py \                          # Execute the Python script using standard Python interpreter
  --root_path "/path/to/your/sunrgbd/data" \       # Directory containing SUN RGB-D dataset (background images, camera calibration, plane data)
  --obj_root_path "/path/to/your/3d/models" \      # Directory containing 3D object models (.obj files) to insert into scenes
  --max_iter 5 \                                   # Generate 5 different variations for each scene in the dataset
  --num_objects_per_scene 6 \                      # Place exactly 6 objects in each generated scene
  --random_seed 12345 \                            # Set random seed for reproducible results across runs
  --ilog 3 \                                       # Environment mapping log parameter (affects lighting calculations)
  --istrength 4                                    # Environment lighting intensity multiplier (higher = brighter lighting)

``` 


## Extra Tools

The repository includes several utility tools for 3D model processing:

- **Resize 3D models**: `resize_models.py`
- **Format conversion**:
  - GLB to OBJ: `convert_glb_to_obj.py`
  - PLY to OBJ: `convert_ply_to_obj.py`




## Output Structure

Output directory structure

```
insertion_ilog2_istren2_context_[timestamp]/
├── annotated_images/        # Visual debugging images with bounding boxes
├── compositional_image/     # Final composite images for training
├── envmap/                  # Environment lighting maps
├── insert_object_log/       # Detailed insertion metadata (JSON)
├── inserted_foreground/     # Rendered 3D objects with transparency
├── label/                   # 3D object detection labels (JSON)
└── yolo_annotations/        # 2D YOLO format annotations (TXT)
```

<table>
<tr>
<th>Directory</th>
<th>Description</th>
<th>Format</th>
</tr>
<tr>
<td><code>compositional_image/</code></td>
<td>Primary training images with inserted 3D objects</td>
<td>PNG</td>
</tr>
<tr>
<td><code>label/</code></td>
<td>3D bounding box annotations</td>
<td>JSON</td>
</tr>
<tr>
<td><code>yolo_annotations/</code></td>
<td>2D bounding box annotations<br><code>class_id center_x center_y width height</code></td>
<td>TXT</td>
</tr>
<tr>
<td><code>insert_object_log/</code></td>
<td>Detailed metadata about object placement and scaling</td>
<td>JSON</td>
</tr>
</table>

## Citation

```bibtex
@inproceedings{3dcopy2024,
  title={3D Copy-Paste: Physically Plausible Object Insertion for Monocular 3D Detection},
  author={[Authors]},
  booktitle={[Conference]},
  year={2024}
}

@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}

@misc{lzqsd,
   title={{Inverse Rendering of Indoor Scene: RIPS 2025 Analog Devices Project}},
   author={zhengqinli},
   howpublished = {\url{https://github.com/lzqsd/InverseRenderingOfIndoorScene}},
   year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.