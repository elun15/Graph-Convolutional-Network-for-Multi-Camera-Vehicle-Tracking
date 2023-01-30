# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash
#conda init
conda activate env_gnn


cd ..
chmod +x main_training.py
#python main_training.py --Mode training
# entrenar BS 100 FPR 5 y 100 quitar el randcrop
python main_training.py --Mode training2
python main_training.py --Mode training2


#python main_training.py --Mode training3

#python main_training.py --Mode training --Options bs_train=64 FPR_alpha=1 add_FPR=True

#python main_training.py --Mode training --Options FPR_alpha=3
#python main_training.py --Mode training --Options bs_train=100 loss_weight=1 loss_weight_custom=False add_FPR=False






