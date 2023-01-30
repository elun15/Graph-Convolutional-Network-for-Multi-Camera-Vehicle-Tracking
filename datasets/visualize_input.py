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


def  process(data_path, input, file,output_dir,mode):
    # Current root directory
    # Original dataset directory
    cameras = os.listdir(data_path)
    # cameras = ['c008']

    c = cameras[0]
    frames_dir = os.path.join(data_path, c, 'img1')
    num_frames = len(os.listdir(frames_dir))



    if not osp.isdir(output_dir):
        os.makedirs(output_dir)

    for f in range(0,num_frames):
        print('Visualizing frame ' + str(f) + '_ ' + str(file))
        fig, axs = plt.subplots(2, int(len(cameras)/2))
        plt.suptitle('Frame ' + str(f+1))

        for cam in range(0,len(cameras)):
                c = cameras[cam]
                [x, y] = np.unravel_index(cam, [2, int(len(cameras)/2)])


                print('Processing ' + c + '_ ' + str(file))
                frames_dir = os.path.join(data_path, c, 'img1')

                img = imread(os.path.join(frames_dir, str(f).zfill(6) + ".jpg"))
                axs[x, y].imshow(img)
                axs[x, y].title.set_text(str(c))
                axs[x,y].axis('off')
                input_dir = os.path.join(data_path, c, input, file + '.txt')

                data = pd.read_csv(input_dir, header=None, sep=",")
                data = data[data.columns[:len(COL_NAMES_AIC)]]
                data.columns = COL_NAMES_AIC

                if mode == 'validation':
                    gt_dir = os.path.join(data_path, c, 'gt', 'gt.txt')

                    gt_data = pd.read_csv(gt_dir, header=None, sep=",")
                    gt_data = gt_data[gt_data.columns[:len(COL_NAMES_AIC)]]
                    gt_data.columns = COL_NAMES_AIC



                data_f = data[data['frame'] == f+1]
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

    mode =  'validation' #'validation'
    scene = 'S02' #'test'
    data_path = '/mnt/rhome/elg/Datasets/AIC21_Track3_MTMC_Tracking/' + mode + '/' + scene +'/'

    input = 'det'
    files = ['mtsc_deepsort_ssd512', 'mtsc_moana_mask_rcnn','mtsc_moana_ssd512', 'mtsc_moana_yolo3','mtsc_tc_mask_rcnn', 'mtsc_tc_ssd512']
    files = ['mtsc_tnt_mask_rcnn','mtsc_tnt_mask_rcnn_roi_zones_statics_filt']
    # files = ['mtsc_tc_yolo3_r']
    files = ['mtsc_tnt_mask_rcnn_roi_zones_statics_filt']
    files = ['mtsc_BUPT21_filtered_roi_filt']
    files = ['mtmc_S02_mtsc_ssd512_tnt_roi_filt_bs2000_cut_T_prun_T_split_T']
    for file in files:
        if mode  == 'validation':
            output_dir = os.path.join('/mnt/rhome/elg/Datasets/AIC21_Track3_MTMC_Tracking/visualizations/', mode, scene,
                                  'vis-img1-gt-' + file)
        if mode == 'test':
            output_dir = os.path.join('/mnt/rhome/elg/Datasets/AIC21_Track3_MTMC_Tracking/visualizations/', mode, scene,
                                      'vis-img1-' + file)

        process(data_path,input, file,output_dir,mode)




