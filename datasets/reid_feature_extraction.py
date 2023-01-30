import os
import time
import shutil
import yaml
import datetime

import matplotlib
# matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import torch
import argparse
from reid.extract_image_feat import ReidFeature
from datasets import dataset, transforms
import pathlib
import pickle as pkl
from torch.utils.data import Dataset
from skimage.io import imread
from PIL import Image


class AIC_dataset_feature_extraction(Dataset):

    def  __init__(self, name, CONFIG, cnn_model, file, dataset_root):

        # self.imageFolderDataset = imageFolderDataset
        self.cnn_model = cnn_model

        # FOR INFERENCE
        transforms_list = ['']
        _, self.transform = transforms.build_transforms(
            CONFIG['CNN_MODEL']['reid_size'][0],
            CONFIG['CNN_MODEL']['reid_size'][1], transforms=transforms_list,
            norm_mean=CONFIG['DATASET']['mean'], norm_std=CONFIG['DATASET']['std'])

        self.transform_list = ''
        self.file = file
        self.path = os.path.join(dataset_root, name)

        self.cameras = os.listdir(self.path)
        self.cameras = [item for item in self.cameras if item[0] is not '.']
        self.cameras.sort()
        self.num_cameras = len(self.cameras)

        self.data_det = pd.DataFrame()
        self.unique_ids_c = []
        self.id_cams_unique_ids_c = []

        for c in self.cameras:
            bboxes_dir = os.path.join(self.path,c, 'img1-bbox-' + file)

            ids = os.listdir(bboxes_dir)
            ids.sort()
            self.unique_ids_c.append(ids)

            self.id_cams_unique_ids_c.append(np.ones(len(ids)) * int(c[1:]))

        self.unique_ids_all = np.concatenate(self.unique_ids_c)
        self.unique_cam_ids_all = np.concatenate(self.id_cams_unique_ids_c)




    def __getitem__(self, index):

        id = self.unique_ids_all[index]
        id_cam = self.unique_cam_ids_all[index]

        bboxes = dict()
        bboxes['id'] = id
        bboxes['cam'] = [id_cam]
        bboxes['bboxes'] = []

        id_path = os.path.join(self.path, 'c' + str(int(id_cam)).zfill(3), 'img1-bbox-' + self.file, str(id).zfill(4))
        bb_imgs_paths = os.listdir(id_path)
        bb_imgs_paths.sort()
        bb_imgs = [] # sort by frame number (the name of the img is the frame)
        size_track = len(bb_imgs_paths)

        if size_track <= 50:
            boxes_sampling = range(0, size_track)  # take all
        else:
            boxes_sampling = (np.linspace(0, size_track-1)).astype(int)

        for b in boxes_sampling:  # process 1 out of each 5 frames
            bb_img = imread(os.path.join(id_path, bb_imgs_paths[b]))
            bb_img = Image.fromarray(bb_img)
            bb_img = self.transform(bb_img)
            bb_imgs.append(bb_img)
        bboxes_stacked = torch.stack(bb_imgs, dim=0)

        bboxes['bboxes'].append(bboxes_stacked)

        return [bboxes]

    def __len__(self):
        return len(self.unique_ids_all)


# Extract and save bbox features
def my_collate(batch):

    bboxes_batches = [item[0] for item in batch]


    return bboxes_batches


def load_reid_model(CONFIG):
    reid_model = ReidFeature(0, CONFIG)

    reid_model.model.cuda()

    return reid_model


global USE_CUDA, CONFIG

USE_CUDA = torch.cuda.is_available()


if __name__ == '__main__':
    dataset_root = './../../../Datasets/AIC21_Track3_MTMC_Tracking/'
    scene= 'validation/S02'
    # files_test= ['mtsc_deepsort_ssd512', 'mtsc_moana_mask_rcnn',
    #          'mtsc_moana_ssd512', 'mtsc_moana_yolo3',
    #          'mtsc_tc_mask_rcnn', 'mtsc_tc_ssd512']
    # files_test = ['mtsc_tnt_mask_rcnn','mtsc_tnt_mask_rcnn_roi_zones_statics_filt']

    files_test = ['mtsc_ssd512_tnt_roi_filt']


    parser = argparse.ArgumentParser(description='REID features extraction from bboxes and saving')
    parser.add_argument('--ConfigPath', metavar='DIR', help='Configuration file path')

    # Decode CONFIG file information
    args = parser.parse_args()
    CONFIG = yaml.safe_load(open(args.ConfigPath, 'r'))

    cnn_model = load_reid_model(CONFIG['CNN_MODEL'])
    cnn_model.model.eval()
    model_path  = pathlib.PurePath(CONFIG['CNN_MODEL']['reid_model'])
    model_name = model_path.name[:-4]

    for file in files_test:
        dataset = AIC_dataset_feature_extraction(scene, CONFIG, cnn_model, file, dataset_root)

        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                             num_workers=6, collate_fn=my_collate,
                                             pin_memory=True)

        if not os.path.exists(os.path.join('./reid_features',scene)):
            os.makedirs(os.path.join('./reid_features',scene))
        tic = time.time()
        with torch.no_grad():
            for i, data in enumerate(loader):
                    print('Iter ' + str(i) + ' / ' + str(len(dataset)/100) + '_ ' + str(file))


                    ########### Data extraction ###########
                    bboxes = data

                    num_ids = len(bboxes)

                    # COMPUTING NODE EMBEDDINGS,
                    for n in range(num_ids):
                        id = bboxes[n]['id']
                        c = bboxes[n]['cam'][0]

                        imgs_bboxes = bboxes[n]['bboxes'][0]
                        print('ID: {}. C: {}. Track with {} images'.format(str(id), str(c), str(imgs_bboxes.shape[0])))

                        imgs_bboxes = imgs_bboxes.cuda()

                        if not os.path.exists(os.path.join('./reid_features', scene,'c'+ str(int(c)).zfill(3))):
                            os.makedirs(os.path.join('./reid_features',scene,'c'+ str(int(c)).zfill(3)))

                        if not os.path.exists(os.path.join('./reid_features',scene, 'c'+ str(int(c)).zfill(3),id)):
                            os.makedirs(os.path.join('./reid_features', scene, 'c'+ str(int(c)).zfill(3),id))

                        with torch.no_grad():
                            bboxes_embeds = cnn_model.model(imgs_bboxes)
                            mean_feature = torch.mean(bboxes_embeds, 0)


                        feature_path = os.path.join('./reid_features',scene,'c'+ str(int(c)).zfill(3),id, file + '_'+ model_name + '.pkl' )

                        with open(feature_path, "wb") as fout:
                            pkl.dump(mean_feature.cpu(), fout, protocol=pkl.HIGHEST_PROTOCOL)

                        fout.close()


        print('elapsed: ' + str(time.time() - tic) + ' secs for ' + str(num_ids) + 'ids')