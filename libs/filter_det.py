import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import shutil
import yaml
import datetime

import matplotlib
# matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset_path = '../datasets/aic19-track1-mtmc/'
mode= 'validation' #'test'
scene = 'S02'
input= 'det'
file = 'det_Efficient8'

COL_NAMES_AIC = ['frame', 'id', 'xmin', 'ymin', 'width', 'height', 'score', 'none', 'none']


###########################################################################
######## SECONF, FILTER BY ZONES    ONLY FOR S06         ##################
###########################################################################


if  scene == 'S02':

    W = 1920
    H = 1080
    ###########################################################################
    ######## FIRST, FILTER BY ROIS PROVIDED BY AIC CHALLENGE ##################
    ###########################################################################
    cameras = os.listdir(os.path.join(dataset_path, mode, scene))
    for c in cameras:
        seq_path = os.path.join(dataset_path, mode, scene, c)
        mtsc_file_path = os.path.join(seq_path, input, file + '.txt')

        roi_path = os.path.join('../ROIs', mode, c, 'roi.jpg')
        det = pd.read_csv(mtsc_file_path, header=None, sep=" ")
        det = det[det.columns[:len(COL_NAMES_AIC)]]
        det.columns = COL_NAMES_AIC[0:len(det.columns)]
        det = det[det['score'] >= 0.25]
        det = det[(det['width'] * det['height']) >= 750]

        roi = plt.imread(roi_path)  # 255 -> OK, 0 -> remove

        if len(roi.shape) == 3:
            roi = roi[:, :, 0]
        roi[roi < 10] = 0  # binarize the ROI
        roi[roi > 240] = 255

        det['ybot'] = (det['ymin'].values + det['height'].values).astype(int)

        ybot = (det['ybot'].values)
        ybot[ybot > 1079] = 1079
        det['ybot'] = ybot

        det['xbot'] = (det['xmin'].values + (det['width'].values / 2)).astype(int)

        xbot = (det['xbot'].values)
        xbot[xbot > 1919] = 1919
        det['xbot'] = xbot


        # det['xbot'].values[(det['xbot'].values) > 1919] = 1919
        det = det[(roi[(det['ybot'].values).astype(int), (det['xbot'].values).astype(int)] == 255)]



        #     if initial_roi == 0 or final_roi == 0:
        #         mtsc = mtsc[mtsc['id'] != i]
        #
        # a = 1
        # mtsc = mtsc[COL_NAMES_AIC]
        np.savetxt(os.path.join(seq_path, input, file + '_025_roi_filt.txt'), det.values, fmt='%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%d', delimiter=',')



