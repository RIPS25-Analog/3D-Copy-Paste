# 3D Copy-Paste

## Overview

This repository provides a pipeline for 3D Copy-Paste data augmentation by inserting realistic 3D objects into the SUN RGB-D dataset. The method generates synthetic training data for 2D object detection by compositing 3D models into real indoor scenes with physically plausible placement and lighting.

## Prerequisites

### Repository Dependencies

Please fork the following repositories under our workspace:

- **[mmdetection3d](https://github.com/open-mmlab/mmdetection3d)** - 3D object detection framework
- **[InverseRenderingOfIndoorScene](https://github.com/[username]/InverseRenderingOfIndoorScene)** - Indoor scene inverse rendering
- **[RGBDPlaneDetection](https://github.com/[username]/RGBDPlaneDetection)** - Plane detection for object placement

### Environment Setup
python and MATLAB is require for this project

please install the following packages
```bash
pip install -r 

```

## Quick Start

> **Note:** For detailed explanations and modifications to the original 3D Copy-Paste implementation, please refer to [original README.md](docs/detailed_guide.md).

### 1. Download SUN RGB-D Dataset

Follow the instructions in the [SUN RGB-D setup guide](docs/sunrgbd_setup.md) to download and prepare the dataset.

### 2. Prepare 3D Models

Ensure your 3D models are in the supported format. See [3D Model Format Requirements](docs/model_formats.md) for details.

### 3. Pre-Data Generation Preparation

```bash
bash Preparation.sh
```

### 4. Generate Synthetic Data

```bash
python generate_3d_copy_paste.py --config configs/default_config.yaml
``` 


explanation on what we changed from original 3D Copy Paste, we have this file:



## Extra Tools

The repository includes several utility tools for 3D model processing:

- **Resize 3D models**: `resize_models.py`
- **Format conversion**:
  - GLB to OBJ: `convert_glb_to_obj.py`
  - PLY to OBJ: `convert_ply_to_obj.py`




## Output Structure

<details>
<summary>Click to expand output directory structure</summary>

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

</details>

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
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.