#!/bin/bash
##
## BSD 3-Clause License
##
## This file is part of the Basalt project.
## https://gitlab.com/VladyslavUsenko/basalt.git
##
## Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
## All rights reserved.
##

set -e
set -x

DATASET_PATH=/home/dm/data/vslam/machine_hall

DATASETS=(MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult)


folder_name=eval_results
# mkdir $folder_name



for d in ${DATASETS[@]}; do
   ./build/basalt_vio --dataset-path  $DATASET_PATH/$d/$d.bag --cam-calib /workspaces/basalt/data/euroc_eucm_calib.json \
        --dataset-type bag --show-gui 0 --config-path /workspaces/basalt/data/euroc_config_schur_ekf.json \
        --result-path $folder_name/vio_$d --save-trajectory tum

   mv trajectory.txt $folder_name/traj_schurvins_$d.txt

    # basalt_mapper --show-gui 0 --cam-calib /usr/etc/basalt/euroc_eucm_calib.json --config-path /usr/etc/basalt/euroc_config.json --marg-data eval_tmp_marg_data \
    #     --result-path $folder_name/mapper_$d

    # basalt_mapper --show-gui 0 --cam-calib /usr/etc/basalt/euroc_eucm_calib.json --config-path /usr/etc/basalt/euroc_config_no_weights.json --marg-data eval_tmp_marg_data \
    #     --result-path $folder_name/mapper_no_weights_$d

    #     basalt_mapper --show-gui 0 --cam-calib /usr/etc/basalt/euroc_eucm_calib.json --config-path /usr/etc/basalt/euroc_config_no_factors.json --marg-data eval_tmp_marg_data \
    #     --result-path $folder_name/mapper_no_factors_$d

    # rm -rf eval_tmp_marg_data
done

#./gen_results.py $folder_name > euroc_results.txt
