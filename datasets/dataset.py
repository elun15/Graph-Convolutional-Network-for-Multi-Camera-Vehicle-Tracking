# import sys
# sys.path.insert(0, './libs/deeppersonreid/torchreid')
# sys.path.insert(0, './libs/deeppersonreid/torchreid/utils')
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.io import imread
from skimage.transform import ProjectiveTransform,warp
import cv2
import utils

# from libs.deeppersonreid.torchreid.data.transforms import build_transforms

# from torchreid.data.transforms import build_transforms
from datasets import transforms
import pathlib
import pickle as pkl
COL_NAMES_AIC = ('frame', 'id', 'xmin', 'ymin', 'width', 'height', 'lost', 'occluded', 'generated','label')

# TRAINING
class AIC_dataset(Dataset):

    def __init__(self, name, mode, CONFIG, cnn_model):

        # self.imageFolderDataset = imageFolderDataset
        self.mode = mode
        self.cnn_model = cnn_model


        if CONFIG['DATASET']['imaug'] is False:
            transforms_list = ['']

        else:
            transforms_list = ['random_resize_crop','color_jitter', 'random_erase']
            # transforms_list = ['color_jitter', 'random_erase']


        self.transform_tr, self.transform_te = transforms.build_transforms(
            CONFIG['CNN_MODEL']['reid_size'][0],
            CONFIG['CNN_MODEL']['reid_size'][1], transforms=transforms_list,
            norm_mean=CONFIG['DATASET']['mean'], norm_std=CONFIG['DATASET']['std'])

        if mode == 'train':

            self.transform = self.transform_tr
            self.transform_list = transforms.RandomHorizontalFlip()
            self.path = os.path.join(CONFIG['DATASET']['root'], mode, name)
            self.input =  CONFIG['DATASET']['input']

        else:
            self.transform_list = ''

            self.path = os.path.join(CONFIG['DATASET']['root'],mode, name)
            self.input = CONFIG['DATASET']['input']
            self.transform = self.transform_te



        self.cameras = os.listdir(self.path)
        self.cameras = [item for item in self.cameras if item[0] is not '.']
        self.cameras.sort()
        self.num_cameras = len(self.cameras)

        self.data_det = pd.DataFrame()
        self.unique_ids_c = []

        for c in self.cameras:
            seq_path = os.path.join(os.path.join(self.path,c))
            detections_file_path = os.path.join(seq_path, self.input, self.input + '.txt')

            det_df = pd.read_csv(detections_file_path, header=None, sep=",")
            det_df = det_df[det_df.columns[:len(COL_NAMES_AIC)]]
            det_df.columns = COL_NAMES_AIC
            det_df['ymax'] = det_df['ymin'].values + det_df['height'].values
            det_df['xmax'] = det_df['xmin'].values + det_df['width'].values
            det_df['label'] = 'CAR'
            det_df['id_cam'] = int(c[1:])
            self.unique_ids_c.append(np.unique(det_df['id'].values))

            self.data_det = self.data_det.append(det_df)

        self.unique_ids_all = np.unique(np.concatenate(self.unique_ids_c))
        # self.unique_ids_all = range(7,17)
        # print('IDs only 7-17')

    def __getitem__(self, index):

        id = self.unique_ids_all[index]


        bboxes_all = dict()
        bboxes_all['id'] = id
        bboxes_all['cam'] = list()
        bboxes_all['bboxes'] = list()


        for c in range(0,self.num_cameras):
            bboxes = []

            if id in self.unique_ids_c[c]:
                # print('Id ' + str(id) + ' .Cam ' + str(c))
                bboxes_all['cam'].append(int(self.cameras[c][1:]))

                id_path = os.path.join(self.path,self.cameras[c],'img1-bbox-'+ self.input, str(id).zfill(4) )
                bb_imgs_paths = os.listdir(id_path)
                bb_imgs_paths.sort() #sort by frame number (the name of the img is the frame)
                size_track = len(bb_imgs_paths)

                if size_track <= 10:
                    boxes_sampling = range(0, size_track)  # take all
                else:
                    boxes_sampling = (np.linspace(0, size_track-1, 10)).astype(int)

                for b in boxes_sampling:
                    # process 1 out of each 5 frames
                    bb_img = imread(os.path.join(id_path, bb_imgs_paths[b]))
                    bb_img = Image.fromarray(bb_img)
                    bb_img = self.transform(bb_img)
                    bboxes.append(bb_img)
                bboxes_stacked =  torch.stack(bboxes, dim=0)

                if bool(self.transform_list) :
                    bboxes_stacked = self.transform_list(bboxes_stacked) #apply same transform to the list of bboxes (in this acse random horizontal flip)

                bboxes_all['bboxes'].append(bboxes_stacked)


        return [bboxes_all]#[bboxes, frames, ids, ids_cam]


    def __len__(self):
        return len(self.unique_ids_all)

# VALIDATION / INFERENCE
class AIC_dataset_inference(Dataset):

    def __init__(self, name, CONFIG, cnn_model):

        # self.imageFolderDataset = imageFolderDataset
        self.cnn_model = cnn_model

               #FOR INFERENCE
        transforms_list = ['']
        _, self.transform_te = transforms.build_transforms(
            CONFIG['CNN_MODEL']['reid_size'][0],
            CONFIG['CNN_MODEL']['reid_size'][1], transforms=transforms_list,
            norm_mean=CONFIG['DATASET']['mean'], norm_std=CONFIG['DATASET']['std'])

        self.transform_list = ''
        self.transform = self.transform_te

        self.input = CONFIG['input_test']
        self.file = CONFIG['file_test']
        self.path = os.path.join(CONFIG['DATASET']['root'], name)

        self.cameras = os.listdir(self.path)
        self.cameras = [item for item in self.cameras if item[0] is not '.']
        self.cameras.sort()
        self.num_cameras = len(self.cameras)

        self.data_det = pd.DataFrame()
        self.unique_ids_c = []
        self.id_cams_unique_ids_c = []

        for c in self.cameras:
            seq_path = os.path.join(os.path.join(self.path,c))
            detections_file_path = os.path.join(seq_path, self.input, self.file + '.txt')

            det_df = pd.read_csv(detections_file_path, header=None, sep=",")
            det_df = det_df[det_df.columns[:len(COL_NAMES_AIC)]]
            det_df.columns = COL_NAMES_AIC
            det_df['ymax'] = det_df['ymin'].values + det_df['height'].values
            det_df['xmax'] = det_df['xmin'].values + det_df['width'].values
            det_df['label'] = 'CAR'
            det_df['id_cam'] = int(c[1:])
            self.unique_ids_c.append(np.unique(det_df['id'].values))
            self.id_cams_unique_ids_c.append(np.ones(len(np.unique(det_df['id'].values))) * int(c[1:]))


            self.data_det = self.data_det.append(det_df)

        self.unique_ids_all =np.concatenate(self.unique_ids_c)

        self.unique_cam_ids_all = np.concatenate(self.id_cams_unique_ids_c)
        a=1



    def __getitem__(self, index):

        id = self.unique_ids_all[index]
        id_cam = self.unique_cam_ids_all[index]

        bboxes = dict()
        bboxes['id'] = id
        bboxes['cam'] = [id_cam]
        bboxes['bboxes'] = []
        bboxes['features'] = []


        id_path = os.path.join(self.path, 'c' + str(int(id_cam)).zfill(3), 'img1-bbox-' + self.file, str(id).zfill(4))
        bb_imgs_paths = os.listdir(id_path)
        bb_imgs_paths.sort()
        bb_imgs =[ ]# sort by frame number (the name of the img is the frame)
        size_track = len(bb_imgs_paths)

        if size_track <= 50:
            boxes_sampling = range(0, size_track) # take all
        else:
            boxes_sampling = (np.linspace(0,size_track-1)).astype(int)

        for b in boxes_sampling:

          # process 1 out of each 5 frames
            bb_img = imread(os.path.join(id_path, bb_imgs_paths[b]))
            bb_img = Image.fromarray(bb_img)
            bb_img = self.transform(bb_img)
            bb_imgs.append(bb_img)
        bboxes_stacked = torch.stack(bb_imgs, dim=0)

        bboxes['bboxes'].append(bboxes_stacked)


        return [bboxes]


    def __len__(self):
        return len(self.unique_ids_all)

#Dataset class for validation , for creating the graph with the precomputed features
class AIC_dataset_inference_precomputed_features(Dataset):

    def __init__(self, name,  CONFIG, cnn_model):

        # self.imageFolderDataset = imageFolderDataset
        self.cnn_model = cnn_model
        self.scenario = name

        model_path = pathlib.PurePath(CONFIG['CNN_MODEL']['reid_model'])
        self.cnn_model_name = model_path.name[:-4]


        self.input = CONFIG['input_test']
        self.file = CONFIG['file_test']
        self.path = os.path.join(CONFIG['DATASET']['root'], name)

        self.cameras = os.listdir(self.path)
        self.cameras = [item for item in self.cameras if item[0] is not '.']
        self.cameras.sort()
        self.num_cameras = len(self.cameras)

        self.data_det = pd.DataFrame()
        unique_ids_c = []
        id_cams_unique_ids_c = []
        frames_life_id_c = []

        for c in self.cameras:
            seq_path = os.path.join(os.path.join(self.path,c))
            detections_file_path = os.path.join(seq_path, self.input, self.file + '.txt')

            det_df = pd.read_csv(detections_file_path, header=None, sep=",")
            det_df = det_df[det_df.columns[:len(COL_NAMES_AIC)]]
            det_df.columns = COL_NAMES_AIC[0:len(det_df.columns)]

            det_df['ymax'] = det_df['ymin'].values + det_df['height'].values
            det_df['xmax'] = det_df['xmin'].values + det_df['width'].values
            det_df['label'] = 'CAR'
            det_df['id_cam'] = int(c[1:])

            unique_ids_c.append(np.unique(det_df['id'].values))
            id_cams_unique_ids_c.append(np.ones(len(np.unique(det_df['id'].values))) * int(c[1:]))
            frames_life_id_c.append(
                [(min(det_df[det_df['id'] == i]['frame'].values), max(det_df[det_df['id'] == i]['frame'].values)) for i
                 in np.unique(det_df['id'].values)])

            self.data_det = self.data_det.append(det_df)

        self.unique_ids_all =np.concatenate(unique_ids_c)

        self.unique_cam_ids_all = np.concatenate(id_cams_unique_ids_c)
        self.frames_life_ids = np.concatenate(frames_life_id_c)


    def __getitem__(self, index):

        id = self.unique_ids_all[index]
        id_cam = self.unique_cam_ids_all[index]


        bboxes = dict()
        bboxes['id'] = id
        bboxes['cam'] = [id_cam]
        bboxes['bboxes'] = []
        bboxes['features'] = []
        bboxes['frames'] = self.frames_life_ids[index, : ]

        feature_path = os.path.join('./reid_features', self.scenario, 'c' + str(int(id_cam)).zfill(3),
                                    str(int(id)).zfill(4), self.file + '_' + self.cnn_model_name + '.pkl')

        if os.path.isfile(feature_path):
            with open(feature_path, "rb") as fout:
                feature_tensor = pkl.load(fout)
        else:
            a=1

        bboxes['features'].append(feature_tensor.numpy())
        # fout.close()



        return [bboxes]


    def __len__(self):
        return len(self.unique_ids_all)


def plot(imgs, with_orig=True, row_title=None, orig_img=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.savefig('test.png')
