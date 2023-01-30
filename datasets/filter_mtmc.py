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

dataset_path = './../../../Datasets/AIC21_Track3_MTMC_Tracking/'
mode= 'validation' #'test'
scene = 'S02'
input= 'mtsc'
file = 'mtsc_deepsort_mask_rcnn_roi_filt' # 'mtsc_deepsort_mask_rcnn'

COL_NAMES_AIC = ['frame', 'id', 'xmin', 'ymin', 'width', 'height', 'lost', 'occluded', 'generated','label']


###########################################################################
######## SECONF, FILTER BY ZONES    ONLY FOR S06         ##################
###########################################################################

if  scene == 'S06':

    ###########################################################################
    ######## FIRST, FILTER BY ROIS PROVIDED BY AIC CHALLENGE ##################
    ###########################################################################
    cameras = os.listdir(os.path.join(dataset_path, mode, scene))
    for c in cameras:
        seq_path = os.path.join(dataset_path, mode, scene, c)
        mtsc_file_path = os.path.join(seq_path, input, file + '.txt')

        roi_path = os.path.join(seq_path, 'roi.jpg')
        mtsc = pd.read_csv(mtsc_file_path, header=None, sep=",")
        mtsc = mtsc[mtsc.columns[:len(COL_NAMES_AIC)]]
        mtsc.columns = COL_NAMES_AIC

        mtsc = mtsc[(mtsc['width'] * mtsc['height']) >= 750]

        roi = plt.imread(roi_path)  # 255 -> OK, 0 -> remove
        roi[roi < 10] = 0  # binarize the ROI
        roi[roi > 240] = 255

        unique_ids = np.unique(mtsc['id'].values)
        for i in unique_ids:

            frame_min = np.min(mtsc[mtsc['id'].values == i]['frame'].values)
            frame_max = np.max(mtsc[mtsc['id'].values == i]['frame'].values)
            mtsc['ybot'] = (mtsc['ymin'].values + mtsc['height'].values)
            mtsc['xbot'] = (mtsc['xmin'].values + (mtsc['width'].values / 2)).astype(int)
            initial_pos = np.asarray(
                [mtsc[(mtsc['id'].values == i) & (mtsc['frame'].values == frame_min)]['ybot'].values[0],
                 mtsc[(mtsc['id'].values == i) & (mtsc['frame'].values == frame_min)]['xbot'].values[0]])
            final_pos = np.asarray(
                [mtsc[(mtsc['id'].values == i) & (mtsc['frame'].values == frame_max)]['ybot'].values[0],
                 mtsc[(mtsc['id'].values == i) & (mtsc['frame'].values == frame_max)]['xbot'].values[0]])

            initial_roi = roi[int(initial_pos[0]), int(initial_pos[1])]
            final_roi = roi[int(final_pos[0]), int(final_pos[1])]

            if initial_roi == 0 or final_roi == 0:
                mtsc = mtsc[mtsc['id'] != i]

        a = 1
        mtsc = mtsc[COL_NAMES_AIC]
        np.savetxt(os.path.join(seq_path, input, file + '_roi_filt.txt'), mtsc.values, fmt='%d', delimiter=',')

    a = 1

    # zone 1 : [255 255 255]
    # zone 2 : [255 0 0]
    # zone 3 : [0 255 0]
    # zone 4 : [0 0 255]

    zone = {
        "[255 255 255]": "W",
        "[255 0 0]": "R",
        "[0 255 0]": "G",
        "[0 0 255]": "B",
        "[0 0 0]": "N"
    }
    cameras = os.listdir(os.path.join(dataset_path,mode,scene))
    for c in cameras:
        seq_path = os.path.join(dataset_path,mode,scene, c)
        mtsc_file_path = os.path.join(seq_path, input, file + '.txt')
        mtsc_file_path = os.path.join(seq_path, input, file + '.txt')
        mask_path = './../img/mask-' + str(int(c[1:])) + '.bmp'
        roi_path = os.path.join(seq_path, 'roi.jpg')
        mtsc = pd.read_csv(mtsc_file_path, header=None, sep=",")
        mtsc = mtsc[mtsc.columns[:len(COL_NAMES_AIC)]]
        mtsc.columns = COL_NAMES_AIC

        ################################################################
        ##                  FILTER MIN SIZE 750 AREA pixels           ##
        ################################################################
        # mtsc = mtsc[(mtsc['width'] >= 61).values & (mtsc['height'] >= 61).values &
        #             (mtsc['xmin'] > 0).values & (mtsc['ymin'] > 0).values]
        mtsc = mtsc[(mtsc['width'] * mtsc['height']) >= 750]


        mask = plt.imread(mask_path)
        mask = mask[:, :, :-1]  # -> divided in 4 zones(4 colors)

        roi = plt.imread(roi_path)  # 255 -> OK, 0 -> remove
        roi[roi < 10] = 0  # binarize the ROI
        roi[roi > 240] = 255

        unique_ids =np.unique(mtsc['id'].values)
        for i in unique_ids:
            frame_min = np.min(mtsc[mtsc['id'].values == i]['frame'].values)
            frame_max = np.max(mtsc[mtsc['id'].values == i]['frame'].values)
            mtsc['ybot'] = (mtsc['ymin'].values + mtsc['height'].values)
            mtsc['xbot'] = (mtsc['xmin'].values + (mtsc['width'].values / 2)).astype(int)
            initial_pos = np.asarray([mtsc[(mtsc['id'].values == i) & (mtsc['frame'].values == frame_min)]['ybot'].values[0], mtsc[(mtsc['id'].values == i) & (mtsc['frame'].values == frame_min)]['xbot'].values[0]])
            final_pos = np.asarray([mtsc[(mtsc['id'].values == i) & (mtsc['frame'].values == frame_max)]['ybot'].values[0], mtsc[(mtsc['id'].values == i) & (mtsc['frame'].values == frame_max)]['xbot'].values[0]])

            initial_roi = roi[initial_pos[0],initial_pos[1]]
            final_roi = roi[final_pos[0],final_pos[1]]

            if initial_roi == 0 or final_roi == 0:
                mtsc = mtsc[mtsc['id'] != i]
            else:
                initial_zone_value = mask[initial_pos[0], initial_pos[1], :]
                final_zone_value = mask[final_pos[0], final_pos[1], :]
                initial_zone = zone[np.array2string(initial_zone_value, formatter={'int_kind':lambda x: "%d" % x})]
                final_zone = zone[np.array2string(final_zone_value, formatter={'int_kind':lambda x: "%d" % x})]

                if (initial_zone == 'W' and final_zone == 'R') or (initial_zone == 'R' and final_zone == 'W'):
                    mtsc = mtsc[mtsc['id'] != i]

                if c == 'c041':
                    if (initial_zone == 'W' and final_zone == 'B') or (initial_zone == 'R' and final_zone == 'B'):
                        mtsc = mtsc[mtsc['id'] != i]

                if c == 'c046':
                    if (initial_zone == 'W' and final_zone == 'G') or (initial_zone == 'R' and final_zone == 'G'):
                        mtsc = mtsc[mtsc['id'] != i]

                # STATICS:
                if initial_zone == final_zone:
                    mtsc = mtsc[mtsc['id'] != i]


        mtsc = mtsc[COL_NAMES_AIC]
        np.savetxt(os.path.join(seq_path, input, file + '_roi_zones_statics_filt.txt'), mtsc.values, fmt='%d',delimiter=',')

        # a=1
        # mtsc = mtsc[COL_NAMES_AIC]
        # np.savetxt(os.path.join(seq_path, input, file + '_roi_filt.txt'), mtsc.values, fmt='%d')

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

        roi_path = os.path.join('./../ROIs',mode, c, 'roi.jpg')
        mtsc = pd.read_csv(mtsc_file_path, header=None, sep=",")
        mtsc = mtsc[mtsc.columns[:len(COL_NAMES_AIC)]]
        mtsc.columns = COL_NAMES_AIC[0:len(mtsc.columns)]

        mtsc = mtsc[(mtsc['width'] * mtsc['height']) >= 750]

        roi = plt.imread(roi_path)  # 255 -> OK, 0 -> remove

        if len(roi.shape) == 3:
            roi = roi[:, :, 0]
        roi[roi < 10] = 0  # binarize the ROI
        roi[roi > 240] = 255

        mtsc['ybot'] = (mtsc['ymin'].values + mtsc['height'].values).astype(int)
        mtsc['ybot'].values[(mtsc['ybot'].values) > 1079] = 1079

        mtsc['xbot'] = (mtsc['xmin'].values + (mtsc['width'].values / 2)).astype(int)
        mtsc['xbot'].values[(mtsc['xbot'].values) > 1919] = 1919
        mtsc = mtsc[(roi[(mtsc['ybot'].values).astype(int), (mtsc['xbot'].values).astype(int)] == 255)]


        np.savetxt(os.path.join(seq_path, input, file + '_roi_filt.txt'), mtsc.values, fmt='%d', delimiter=',')

if  scene == 'S05':

    W = 1920
    H = 1080
    ###########################################################################
    ######## FIRST, FILTER BY ROIS PROVIDED BY AIC CHALLENGE ##################
    ###########################################################################
    cameras = os.listdir(os.path.join(dataset_path, mode, scene))
    for c in cameras:
        seq_path = os.path.join(dataset_path, mode, scene, c)
        mtsc_file_path = os.path.join(seq_path, input, file + '.txt')

        roi_path = os.path.join('./../ROIs',mode, c, 'roi.jpg')
        mtsc = pd.read_csv(mtsc_file_path, header=None, sep=",")
        mtsc = mtsc[mtsc.columns[:len(COL_NAMES_AIC)]]
        mtsc.columns = COL_NAMES_AIC[0:len(mtsc.columns)]

        mtsc = mtsc[(mtsc['width'] * mtsc['height']) >= 750]

        roi = plt.imread(roi_path)  # 255 -> OK, 0 -> remove

        if len(roi.shape) == 3:
            roi = roi[:, :, 0]
        roi[roi < 10] = 0  # binarize the ROI
        roi[roi > 240] = 255

        mtsc['ybot'] = (mtsc['ymin'].values + mtsc['height'].values).astype(int)
        mtsc['ybot'].values[(mtsc['ybot'].values) > 1079] = 1079

        mtsc['xbot'] = (mtsc['xmin'].values + (mtsc['width'].values / 2)).astype(int)
        mtsc['xbot'].values[(mtsc['xbot'].values) > 1919] = 1919
        mtsc = mtsc[(roi[(mtsc['ybot'].values).astype(int), (mtsc['xbot'].values).astype(int)] == 255)]


        np.savetxt(os.path.join(seq_path, input, file + '_roi_filt.txt'), mtsc.values, fmt='%d', delimiter=',')



