'''
################################
#    Pre-process  data
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

COL_NAMES_AIC = ('frame', 'id', 'xmin', 'ymin', 'width', 'height', 'lost', 'occluded', 'generated','label')


def  process(data_path, input, file):
    COL_NAMES_AIC = ('frame', 'id', 'xmin', 'ymin', 'width', 'height', 'lost', 'occluded', 'generated', 'label')

    if file == 'mtsc_ssd512_tnt_roi_filt' or file == 'mtsc_tnt_effdet8_roi_filt' or file == 'mtsc_tnt_effdet8_025_roi_filt':
        COL_NAMES_AIC = ('frame', 'id', 'xmin', 'ymin', 'width', 'height', 'lost', 'occluded', 'generated')

    # Current root directory
    # Original dataset directory
    cameras = os.listdir(data_path)
    for c in cameras:

            tStart = time.time()
            print('Processing ' + c + '_ ' + str(file))

            frames_dir = os.path.join(data_path, c, 'img1')
            num_frames = len(os.listdir(frames_dir))
            input_dir = os.path.join(data_path, c, input, file + '.txt')

            output_dir = os.path.join(data_path, c, 'img1-bbox-' + file)
            # if not os.path.exists(output_dir):
            # os.makedirs(output_dir)

            data = pd.read_csv(input_dir, header=None, sep=",")
            data = data[data.columns[:len(COL_NAMES_AIC)]]
            data.columns = COL_NAMES_AIC
            frames_unique = np.unique(data['frame'].values)

            # ids_unique = np.unique(data['id'].values)
            # length_ids = np.asarray([len(data[data['id'] == i]) for i in ids_unique])

            for f in frames_unique:
                print('Processing frame ' + str(f) + '_ ' + str(file))
                data_f = data[data['frame']== f]
                img = imread(os.path.join(frames_dir, str(f-1).zfill(6) + ".jpg"))
                for row in range(0,data_f.shape[0]):
                    obj_id = int(data_f.iloc[row]['id'])
                    out_obj_path = osp.join(output_dir, str(obj_id).zfill(4))

                    height, width = img.shape[:2]

                    left =int(data_f.iloc[row]['xmin'])
                    top = int(data_f.iloc[row]['ymin'])
                    w = int(data_f.iloc[row]['width'])
                    h = int(data_f.iloc[row]['height'])

                    right = left + w
                    bot = top + h

                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0

                    if right > width:
                        right = width
                    if bot > height:
                        bot = height

                    crop_img = img[top: bot, left:right]
                    if crop_img.size == 0:
                        data = data.drop(data[(data['frame'] == f).values & (data['id'] == obj_id).values].index.values[0])
                    else:
                        if not osp.isdir(out_obj_path):
                            os.makedirs(out_obj_path)
                        out_path = osp.join(out_obj_path, str(f).zfill(5) + ".jpg")
                        imsave(out_path, crop_img)

            data.to_csv(input_dir,header=None, sep=",", index=False)

            tEnd = time.time()
            print("It cost %f sec" % (tEnd - tStart))


if __name__ == '__main__':
    data_path = './../../../Datasets/AIC21_Track3_MTMC_Tracking/validation/S02/'
    input = 'mtsc'
    #  example files = ['mtsc_deepsort_ssd512', 'mtsc_moana_mask_rcnn',
    #               'mtsc_moana_ssd512', 'mtsc_moana_yolo3',
    #               'mtsc_tc_mask_rcnn', 'mtsc_tc_ssd512', 'mtsc_BUPT21_filtered_roi_filt']

    files = ['mtsc_deepsort_mask_rcnn_roi_filt']


    for file in files:
        process(data_path,input, file)




