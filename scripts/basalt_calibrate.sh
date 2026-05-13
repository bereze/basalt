#!/bin/bash

# basalt_calibrate run script with editable parameters
DATASET_PATH="/home/dm/data/insight/calib_data/rosbag2_20250826_233325"
DATASET_TYPE="bag2"
APRILGRID="data/aprilgrid_6x6.json"
RESULT_PATH="/home/dm/data/insight/calib_data/rosbag2_20250826_233325/output_pinhole"
CAM_TYPES=("pinhole" "pinhole")

./build/basalt_calibrate \
  --dataset-path "${DATASET_PATH}" \
  --dataset-type "${DATASET_TYPE}" \
  --aprilgrid "${APRILGRID}" \
  --result-path "${RESULT_PATH}" \
  --cam-types "${CAM_TYPES[@]}"
