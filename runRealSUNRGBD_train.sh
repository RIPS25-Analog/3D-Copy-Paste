# TEST
python3 testReal.py --cuda --dataRoot /home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/mmdetection3d/data/sunrgbd/sunrgbd_trainval/image_resize_640480 --imList imList_SUNRGBD_train1.txt \
    --testRoot RealSUNRGBD --isLight --isBS --level 2 \
    --experiment0 check_cascade0_w320_h240 --nepoch0 14 \
    --experimentLight0 check_cascadeLight0_sg12_offset1.0 --nepochLight0 10 \
    --experimentBS0 checkBs_cascade0_w320_h240 \
    --experiment1 check_cascade1_w320_h240 --nepoch1 7 \
    --experimentLight1 check_cascadeLight1_sg12_offset1.0 --nepochLight1 10 \
    --experimentBS1 checkBs_cascade1_w320_h240 \
 /home/coraguo/RIPS25-AnalogDevices-ObjectDetection/src/3d_C_P/3D-Copy-Paste/runRealSUNRGBD_train.sh