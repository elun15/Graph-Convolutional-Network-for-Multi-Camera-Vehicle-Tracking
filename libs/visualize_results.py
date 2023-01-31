'''
################################
#    Pre-process EPFL data
#   Extract bbox images from an input file

################################
'''


import time
import cv2
import os
import os.path as osp
import pandas as pd
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from matplotlib import patches

COL_NAMES_AIC = ('frame', 'id', 'xmin', 'ymin', 'width', 'height', 'lost', 'occluded', 'generated','label')
COL_NAMES_MTMC = ('cam',  'id', 'frame', 'xmin', 'ymin', 'width', 'height')


def  process(mtmc_path,data_path, output_dir,mode):
    # Current root directory
    # Original dataset directory
    cameras = os.listdir(data_path)
    # cameras = ['c008']

    c = cameras[0]
    frames_dir = os.path.join(data_path, c, 'img1')
    num_frames = len(os.listdir(frames_dir))

    data_mtmc = pd.read_csv(mtmc_path+'.txt', header=None, sep=" ")
    data_mtmc = data_mtmc[data_mtmc.columns[:len(COL_NAMES_MTMC)]]
    data_mtmc.columns = COL_NAMES_MTMC

    if not osp.isdir(output_dir):
        os.makedirs(output_dir)

    for f in range(0,num_frames):
        print('Visualizing frame ' + str(f))
        fig, axs = plt.subplots(2, int(len(cameras)/2))
        plt.suptitle('Frame ' + str(f+1))

        for cam in range(0,len(cameras)):
                c = cameras[cam]
                [x, y] = np.unravel_index(cam, [2, int(len(cameras)/2)])


                print('Processing ' + c )
                frames_dir = os.path.join(data_path, c, 'img1')

                img = imread(os.path.join(frames_dir, str(f).zfill(6) + ".jpg"))
                axs[x, y].imshow(img)
                axs[x, y].title.set_text(str(c))
                axs[x,y].axis('off')

                data_mtmc_c = data_mtmc[data_mtmc['cam'] == int(str(c[-2:]))]
                # data_mtmc = data_mtmc[data_mtmc.columns[:len(COL_NAMES_MTMC)]]
                # data_mtmc.columns = COL_NAMES_MTMC

                if mode == 'validation':
                    gt_dir = os.path.join(data_path, c, 'gt', 'gt.txt')

                    gt_data = pd.read_csv(gt_dir, header=None, sep=",")
                    gt_data = gt_data[gt_data.columns[:len(COL_NAMES_AIC)]]
                    gt_data.columns = COL_NAMES_AIC



                data_f = data_mtmc_c[data_mtmc_c['frame'] == f+1]
                if mode == 'validation':

                    gt_f = gt_data[gt_data['frame'] == f+1]


                for row in range(0, data_f.shape[0]):
                    obj_id = int(data_f.iloc[row]['id'])
                    left = int(data_f.iloc[row]['xmin'])
                    top = int(data_f.iloc[row]['ymin'])
                    w = int(data_f.iloc[row]['width'])
                    h = int(data_f.iloc[row]['height'])
                    rect = patches.Rectangle((left, top), w, h, edgecolor='red', facecolor="none",lw = 0.75)
                    axs[x,y].add_patch(rect)
                    axs[x,y].text(left +w+10  , top+h +10, str(obj_id),fontsize=7,color= 'red')

                if mode == 'validation':
                    for row in range(0, gt_f.shape[0]):
                        obj_id = int(gt_f.iloc[row]['id'])
                        left = int(gt_f.iloc[row]['xmin'])
                        top = int(gt_f.iloc[row]['ymin'])
                        w = int(gt_f.iloc[row]['width'])
                        h = int(gt_f.iloc[row]['height'])
                        rect = patches.Rectangle((left, top), w, h, edgecolor='lime', facecolor="none", lw = 0.65)
                        axs[x,y].add_patch(rect)
                        axs[x,y].text(left -10 , top-10, str(obj_id),fontsize=5,color= 'lime')
                a=1
        # plt.show(block=False)
        # plt.waitforbuttonpress()
        out_img_path = osp.join(output_dir, str(f).zfill(6)+'.png')
        plt.savefig(out_img_path, dpi=300)
        plt.close()

if __name__ == '__main__':

    mode = 'validation'
    model =  'tr_S01-S03-S04_val_S02_resnet101_ibn_a_2_weight_custom_SGD_lr_0.01_BS_100_150_L_1_RandCrop_0510_0713__2022-05-10 12:51:17'
    scene = 'S02' #'test'
    file_mtmc = 'mtmc_S02_mtsc_ssd512_tnt_roi_filt_bs2000_cut_T_prun_T_split_T'
    mtmc_path = '/../results/' + model
    data_path = '/../datasets/aic19-track1-mtmc/' + mode + '/' + scene +'/'

    if mode  == 'validation':
        output_dir = os.path.join(mtmc_path, 'vis-img1-gt-' + file_mtmc)


    process(mtmc_path + '/'+ file_mtmc,data_path, output_dir,mode)




