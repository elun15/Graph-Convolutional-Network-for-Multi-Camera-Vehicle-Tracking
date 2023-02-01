'''
################################
#    Pre-process  data
#    Extract frame images from .avi
################################
'''

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def  process(data_path):
    # Current root directory


    # Original dataset directory


    cameras = os.listdir(data_path)
    for c in cameras:

            tStart = time.time()
            print('Processing ' + c)

            vdo_dir = os.path.join(data_path, c, 'vdo.avi')
            # vdo_dir = os.path.join(data_path, c, c + '.mp4')

            video = cv2.VideoCapture(vdo_dir)

            num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = video.get(cv2.CAP_PROP_FPS)
            h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

            blank_image = np.zeros((h, w, 3), np.uint8)

            output_dir = os.path.join(data_path, c, 'img1')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)


            frame_counter = 0

            # Read video file and save image frames
            while video.isOpened():

                ret, frame = video.read()
                frame_name = os.path.join(output_dir, str(frame_counter).zfill(6) + ".jpg")
                frame_counter += 1

                # print(video.get(cv2.CAP_PROP_POS_FRAMES))

                if not ret:
                    print("End of video file.")
                    a = 1
                    break
                cv2.imwrite(frame_name, frame)


            tEnd = time.time()
            print("It cost %f sec" % (tEnd - tStart))


if __name__ == '__main__':
    data_path = './../datasets/AIC21_Track3_MTMC_Tracking/validation/S02/'
    process(data_path)




