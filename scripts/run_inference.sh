# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash
#conda init
conda activate env_gnn

cd ..
chmod +x main.py

#python main.py --ConfigPath "./config/config_inference.yaml"

python main.py --Model "tr_S01-S03-S04_val_S02_resnet101_ibn_a_2_weight_None_BS_64__Adam_custom_batch_weight_2022-03-15 22:09:33" --Options data_test=validation/S02 bs_test=900 input_test=gt file_test=gt CUTTING=True PRUNING=False SPLITTING=True

