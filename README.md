

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

To create and setup the Anaconda Envirmorent run the following terminal command from the repository folder:
```
conda env create -f env_gnn.yml
conda activate env_gnn
```

**3. Download AIC19 dataset**

This repo is evaluated on  the <u>AI City Challenge 2019 dataset</u>. Trained in S01 and tested in S02 scenarios.
Download the data from [https://www.aicitychallenge.org/track1-download/](https://www.aicitychallenge.org/track1-download/).


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



**5. Run** 
```
python ./libs/preprocess_EPFL.py
```
 in order to extract frame images. 

**6. Ground-truth** 

 The EPFL GT (we already provide it, no need to download it)  can be found at [https://bitbucket.org/merayxu/multiview-object-tracking-dataset/src/master/](https://bitbucket.org/merayxu/multiview-object-tracking-dataset/src/master/). 


**7. Download pre-trained REID models**

  Download the pre-trained REID models from https://1drv.ms/u/s!AufOLvb5OB5fhx0os9hCDdkFfT6l?e=roljmV  , unzip the 4 folders and place them under *./trained_models/*

**8. Download  a pre-trained GNN-CCA model**

We provide the weights of the GNN trained on the S1 set (see paper for detailes).
Download the pre-trained weights from https://1drv.ms/u/s!AufOLvb5OB5fhx7O9KIJDqKLj8Uu?e=hbyR7T and place the folder *GNN_S1_Resnet50MCD_SGD0005_cosine20_BS64_BCE_all_step_BNcls_L4_2021-11-10 19:01:49* under *./results/* folder.

**9. Inference Running**

To inference the previous model run:
```
python main.py --ConfigPath "config/config_inference.yaml"
```
**10. Training**

For training run:
```
python main_training.py --ConfigPath "config/config_training.yaml"
```


## Citation

If you find this code and work useful, please consider citing:
```
@ARTICLE{9893862,
  author={Luna, Elena and SanMiguel, Juan C. and Martínez, José M. and Carballeira, Pablo},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Graph Neural Networks for Cross-Camera Data Association}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3207223}}
}
```

