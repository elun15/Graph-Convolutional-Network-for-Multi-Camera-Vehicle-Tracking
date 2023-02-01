



# Graph-Convolutional-Network-for-Multi-Camera-Vehicle-Tracking

ArXiv paper:  https://arxiv.org/pdf/2211.15538.pdf

## Setup & Running
**Requirements**

The repository has been tested in the following software.
* Ubuntu 16.04
* Python 3.6
* Anaconda
* Pycharm

**1. Clone repository**

```
git clone https://github.com/elun15/Graph-Convolutional-Network-for-Multi-Camera-Vehicle-Tracking.git
```

**2. Anaconda environment**

To create and setup the Anaconda Environment run the following terminal command from the repository folder:
```
conda env create -f env_gnn.yml
conda activate env_gnn
```

**3. Download AIC19 dataset**

This repo is evaluated on  the <u>AI City Challenge 2019 dataset</u>. Trained in S01 and tested in S02 scenarios.
Download the data from [https://www.aicitychallenge.org/track1-download/](https://www.aicitychallenge.org/track1-download/) and place the it under *./datasets/*.


**4. Download the vehicle ReID code**
From https://github.com/LCFractal/AIC21-MTMC, download the ReID model (resnet101_ibn_a_2.pth) and place it under *reid/reid_model/* like:

>   * reid
>     * reid_model 
>       * resnet101_ibn_a_2.pth

 
**5. Prepare AIC19 dataset**

Once downloaded the folder *aic19-track1-mtmc/*, preprocess the data by running the following (please note that the dataset path must be modified accordingly):

To extract frames' images from .avi videos:                                                                
```
python ./datasets/preprocess_AIC.py
```
To filter MOT by ROIs:
```
python ./datasets/filter_mtmc.py
```
**6. Extract and save ReID features (optional, to save computational time)**

```
python ./datasets/reid_feature_extraction.py --ConfigPath ./config/config_feature_extraction.yaml
```

**7. Run Inference** 
We provide the trained weights in HERE. Donwload it and place it under *./results/*.
The inference of the model can be done running:
```
python main.py --Model "tr_S01-S03-S04_val_S02_resnet101_ibn_a_2_weight_custom_SGD_lr_0.01_BS_100_150_L_1_1FPR__2022-04-27 17-01-51" --Options data_test=validation/S02 input_test=mtsc file_test=mtsc_ssd512_tnt_roi_filt bs_test=2000 CUTTING=True PRUNING=True SPLITTING=True pre_computed_feats=True
```

**8. Training**

For training run:
```
python main_training.py --Mode training
```

(*./config/config_training.yaml* will be considered)


## Citation

If you find this code and work useful, please consider citing:
```
@article{luna2022graph,
  title={Graph Convolutional Network for Multi-Target Multi-Camera Vehicle Tracking},
  author={Luna, Elena and Miguel, Juan Carlos San and Mart{\'\i}nez, Jos{\'e} Mar{\'\i}a and Escudero-Vi{\~n}olo, Marcos},
  journal={arXiv preprint arXiv:2211.15538},
  year={2022}
}
```

