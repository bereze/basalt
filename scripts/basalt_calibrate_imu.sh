#!/bin/bash

# basalt_calibrate_imu run script with editable parameters
DATASET_PATH="/home/dm/data/insight/calib_data/rosbag2_20250826_233325"
DATASET_TYPE="bag2"
APRILGRID="data/aprilgrid_6x6.json"
RESULT_PATH="/home/dm/data/insight/calib_data/rosbag2_20250826_233325/output_pinhole"
GYRO_NOISE_STD="0.001979209102026288"
ACCEL_NOISE_STD="0.015002460848908"
GYRO_BIAS_STD="4.255409147145483e-05"
ACCEL_BIAS_STD="5.297722385752815e-04"

./build/basalt_calibrate_imu \
  --dataset-path "${DATASET_PATH}" \
  --dataset-type "${DATASET_TYPE}" \
  --aprilgrid "${APRILGRID}" \
  --result-path "${RESULT_PATH}" \
  --gyro-noise-std "${GYRO_NOISE_STD}" \
  --accel-noise-std "${ACCEL_NOISE_STD}" \
  --gyro-bias-std "${GYRO_BIAS_STD}" \
  --accel-bias-std "${ACCEL_BIAS_STD}"
