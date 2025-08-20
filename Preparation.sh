set -e

# Update these paths to your local directories
CP_3D_DIR="path/to/3D-Copy-Paste"
mmdetection3d_dir="path/to/mmdetection3d"
RGBDPlaneDetection_dir="path/to/RGBDPlaneDetection"
InverseRendering_dir="path/to/InverseRendering"
3D_Models_dir="path/to/3D-Models"
Background_dir="path/to/SUNRGBD_Dataset"
MATLAB_dir="path/to/MATLAB"

# Step 1: Create directory structure and move files
cd "$mmdetection3d_dir"
mkdir -p ./data/sunrgbd/OFFICIAL_SUNRGBD
mv "$Background_dir/SUNRGBD.zip" "./data/sunrgbd/OFFICIAL_SUNRGBD/" 2>/dev/null || true
mv "$Background_dir/SUNRGBDMeta2DBB_v2.mat" "./data/sunrgbd/OFFICIAL_SUNRGBD/" 2>/dev/null || true
mv "$Background_dir/SUNRGBDMeta3DBB_v2.mat" "./data/sunrgbd/OFFICIAL_SUNRGBD/" 2>/dev/null || true
mv "$Background_dir/SUNRGBDtoolbox.zip" "./data/sunrgbd/OFFICIAL_SUNRGBD/" 2>/dev/null || true

cd ./data/sunrgbd/OFFICIAL_SUNRGBD
[ -f "SUNRGBD.zip" ] && unzip -q SUNRGBD.zip
[ -f "SUNRGBDtoolbox.zip" ] && unzip -q SUNRGBDtoolbox.zip

# Step 2: Extract point clouds and annotations using MATLAB
cd "$MATLAB_dir"
matlab -nosplash -nodesktop -r 'extract_split; quit;' > /dev/null 2>&1
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v2; quit;' > /dev/null 2>&1
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v1; quit;' > /dev/null 2>&1

# Step 3: Generate training data
cd "$mmdetection3d_dir"
python tools/create_data.py sunrgbd \
    --root-path ./data/sunrgbd \
    --out-dir ./data/sunrgbd \
    --extra-tag sunrgbd > /dev/null 2>&1

echo "Resizing images to 640x480..."

# Step 4: Resize images
cd "$CP_3D_DIR/src"
python plane_detection_1_resize_to_640480.py --source_root "$mmdetection3d_dir/data/sunrgbd/sunrgbd_trainval"

echo "Running plane detection..."

# Step 5: Plane detection
cd "$RGBDPlaneDetection_dir"
target_path="$mmdetection3d_dir/data/sunrgbd/sunrgbd_trainval/plane"
for i in $(seq 1 1000); do
    mkdir -p "$target_path/$(printf "%06d" $i)"
    rgb="$mmdetection3d_dir/data/sunrgbd/sunrgbd_trainval/image_resize_640480/$(printf "%06d" $i).png"
    depth="$mmdetection3d_dir/data/sunrgbd/sunrgbd_trainval/raw_depth_resize_640480/$(printf "%06d" $i).png"
    output_dir="$mmdetection3d_dir/data/sunrgbd/sunrgbd_trainval/plane/$(printf "%06d" $i)/"
    ./build/RGBDPlaneDetection -o "$rgb" "$depth" "$output_dir" > /dev/null 2>&1
done

echo "Analyzing planes..."

# Step 6: Plane analysis
cd "$CP_3D_DIR"
python plane_detection_2_plane_analysis.py --source_root "$mmdetection3d_dir/data/sunrgbd/sunrgbd_trainval"
python plane_detection_3_floor_detection_and_filter.py --source_root "$mmdetection3d_dir/data/sunrgbd/sunrgbd_trainval"

echo "Running inverse rendering..."

# Step 7: Inverse rendering
cd "$InverseRendering_dir"
dataroot_path="$mmdetection3d_dir/data/sunrgbd/sunrgbd_trainval/image_resize_640480"
python3 testReal.py --cuda \
    --dataRoot "$dataroot_path" \
    --imList image_list.txt \
    --testRoot NYU_cascade1 \
    --isLight --isBS --level 2 \
    --experiment0 check_cascadeNYU0 --nepoch0 2 \
    --experimentLight0 check_cascadeLight0_sg12_offset1 --nepochLight0 10 \
    --experimentBS0 checkBs_cascade0_w320_h240 \
    --experiment1 check_cascadeNYU1 --nepoch1 3 \
    --experimentLight1 check_cascadeLight1_sg12_offset1 --nepochLight1 10 \
    --experimentBS1 checkBs_cascade1_w320_h240