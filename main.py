# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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

import torch
import argparse

from torch.utils.data import DataLoader,Dataset


from datasets import dataset
from models.mpn import MOTMPNet

import utils

from reid.extract_image_feat import ReidFeature

from inference import inference, inference_precomputed_features

import pathlib
from libs import getConfiguration
from eval import eval


from ptflops import get_model_complexity_info


def load_reid_model(CONFIG):
    reid_model = ReidFeature(0, CONFIG)
    #     model = ReidFeature(gpu_id.get(), _cfg)
    # reid_feat_numpy = model.extract(chunk_list[0])

    reid_model.model.cuda()

    return reid_model

def load_model_mpn(CONFIG,weights_path=None):
    if weights_path is None:
        model = MOTMPNet(CONFIG['GRAPH_NET_PARAMS'], None,CONFIG['CNN_MODEL']['arch']).cuda()
    else:
        model = MOTMPNet(CONFIG['GRAPH_NET_PARAMS'], None,CONFIG['CNN_MODEL']['arch']).cuda()

        model = utils.load_pretrained_weights(model, weights_path)
    return model

def my_collate(batch):

    bboxes_batches = [item[0] for item in batch]


    return bboxes_batches


global USE_CUDA, CONFIG

USE_CUDA = torch.cuda.is_available()

date = date = str(time.localtime().tm_year) + '-' + str(time.localtime().tm_mon).zfill(2) + '-' + str(
    time.localtime().tm_mday).zfill(2) + \
       ' ' + str(time.localtime().tm_hour).zfill(2) + ':' + str(time.localtime().tm_min).zfill(2) + ':' + str(
    time.localtime().tm_sec).zfill(2)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--Model', metavar='DIR', required=True, help='Model file path')
parser.add_argument('--Options', metavar='DIR', nargs='*', help='an integer for the accumulator')


# Decode CONFIG file information
args = parser.parse_args()
CONFIG = getConfiguration.getValidationConfigurationWithOptions(args)

# Decode CONFIG file information
results_path = os.path.join('./results', args.Model)


cnn_model = load_reid_model(CONFIG['CNN_MODEL'])
# macs, params = get_model_complexity_info(cnn_model.model, (3, 384, 384), as_strings=True,print_per_layer_stat=True, verbose=True)

if type(CONFIG['pre_computed_feats']) is str:
    if CONFIG['pre_computed_feats'] == 'True':
        CONFIG['pre_computed_feats'] = True
    else:
        CONFIG['pre_computed_feats'] = False

if CONFIG['pre_computed_feats']:
    dataset = dataset.AIC_dataset_inference_precomputed_features(CONFIG['data_test'], CONFIG, cnn_model)
else:
    dataset = dataset.AIC_dataset_inference(CONFIG['data_test'], CONFIG, cnn_model)



loader = torch.utils.data.DataLoader(dataset, batch_size=int(CONFIG['bs_test']), shuffle=False,
                                       num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], collate_fn=my_collate,pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])



info_label = str(pathlib.PurePath(CONFIG['data_test']).name) + '_' + CONFIG['file_test'] + '_bs'+CONFIG['bs_test'] +  '_cut_' + str(CONFIG['CUTTING'])[0] + '_prun_' +  str(CONFIG['PRUNING'])[0]+ '_split_' + str(CONFIG['SPLITTING'][0])
model_path  = pathlib.PurePath(CONFIG['CNN_MODEL']['reid_model'])
cnn_model_name = model_path.name[:-4]


mpn_model = load_model_mpn(CONFIG, os.path.join(results_path,'models','best.pth.tar'))
mpn_model.cuda()
mpn_model.eval()
# macs, params = get_model_complexity_info(mpn_model.encoder.edge_mlp, (1,2), as_strings=True,print_per_layer_stat=True, verbose=True)
val_loss_in_history = []
prec0 = []
prec1 = []

epoch = 0

if CONFIG['pre_computed_feats']:
    P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index, mutual_index, homogeneity, completeness, v_measure, prec0, prec1, data_tracking = \
        inference_precomputed_features(CONFIG, loader, cnn_model, mpn_model)
else:
    P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index,  mutual_index, homogeneity, completeness, v_measure, prec0, prec1,data_tracking = \
    inference(CONFIG, loader, cnn_model, mpn_model)


# np.savetxt(os.path.join(results_path,'mtmc_' + info_label + '.txt'), data_tracking.values, fmt='%d')
#
# if CONFIG['input_test'] == 'gt':
#     P = torch.mean(torch.stack(P_list))
#     R = torch.mean(torch.stack(R_list))
#     F = torch.mean(torch.stack(F_list))
#     TP =(torch.stack(TP_list))
#     FP = (torch.stack(FP_list))
#     FN = (torch.stack(FN_list))
#     TN = (torch.stack(TN_list))
#     RI = np.mean(np.asarray(rand_index))
#     MI = np.mean(np.asarray(mutual_index))
#     hom = np.mean(np.asarray(homogeneity))
#     com = np.mean(np.asarray(completeness))
#     v = np.mean(np.asarray(v_measure))
#     prec0 = torch.mean(torch.stack(prec0))
#     prec1 = torch.mean(torch.stack(prec1))
#
#     f = open(results_path + '/clustering_' + info_label +'.txt', "w")
#     f.write('P= ' + str(P.item()) + '\n')
#     f.write('R= ' + str(R.item()) + '\n')
#     f.write('F= ' + str(F.item())+ '\n')
#     f.write('TP= ' + str(TP.item())+ '\n')
#     f.write('FP= ' + str(FP.item())+ '\n')
#     f.write('FN= ' + str(FN.item())+ '\n')
#     f.write('TN= ' + str(TN.item())+ '\n')
#     f.write('Rand index mean = ' + str(RI) + '\n')
#     f.write('Mutual index mean = ' + str(MI)+ '\n')
#     f.write('homogeneity mean = ' + str(hom)+ '\n')
#     f.write('completeness mean = ' + str(com)+ '\n')
#     f.write('v_measure mean = ' + str(v)+ '\n')
#     f.write('Mean prec 0 = ' + str(prec0.item())+ '\n')
#     f.write('Mean prec 1 = ' + str(prec1.item())+ '\n')
#
#     f.close()
#
#
#     print('P= '+ str(P.item()))
#     print('R= '+ str(R.item()))
#     print('F= '+ str(F.item()))
#     print('TP= ' + str(TP.item()))
#     print('FP= ' + str(FP.item()))
#     print('FN= ' + str(FN.item()))
#     print('TN= '+ str(TN.item()))
#     print('Rand index mean = ' + str(RI))
#     print( 'Mutual index mean = ' + str(MI) )
#     print( 'homogeneity mean = ' + str(hom) )
#     print( 'completeness mean = ' + str(com) )
#     print( 'v_measure mean = ' + str(v) )
#     print('Mean prec 0 = ' + str(prec0.item()) )
#     print('Mean prec 1 = ' + str(prec1.item()) )
#
# if CONFIG['data_test'] == 'validation/S02':
#     test = eval.readData('./eval/ground_truth_S02.txt')
#     pred = data_tracking.values
#     # pred = eval.readData(os.path.join(results_path, 'mtmc_' + info_label + '.txt'))
#
#
#     flag_eval_center = True
#     if flag_eval_center:
#         W = 1920
#         H = 1080
#         p = 0.1
#
#         ###  GT
#         cond1 = (W * p <= test['X'].values + (test['Width'].values / 2))
#         cond2 = test['X'].values + (test['Width'].values / 2) <= W - W * p
#
#         idx_x = np.logical_and(cond1, cond2)
#
#         cond1 = (H * p <= test['Y'].values + test['Height'].values)
#         cond2 = test['Y'].values + test['Height'].values <= H - H * p
#         idx_h = np.logical_and(cond1, cond2)
#
#         idx = np.logical_and(idx_x, idx_h)
#
#         test = test[idx]
#
#         ###  PREDICTIONS
#
#         cond1 = (W * p <= pred['X'].values + (pred['Width'].values / 2))
#         cond2 = pred['X'].values + (pred['Width'].values / 2) <= W - W * p
#
#         idx_x = np.logical_and(cond1, cond2)
#
#         cond1 = (H * p <= pred['Y'].values + pred['Height'].values)
#         cond2 = pred['Y'].values + pred['Height'].values <= H - H * p
#         idx_h = np.logical_and(cond1, cond2)
#
#         idx = np.logical_and(idx_x, idx_h)
#
#         pred = pred[idx]
#
#
#     summary,th = eval.eval(test, pred, mread=False, dstype=os.path.split(CONFIG['data_test'])[0], roidir='ROIs', th=0.75)
#     eval.print_results(summary, mread=False)
#
#
#     if flag_eval_center:
#         summary.to_excel(os.path.join(results_path,  'center_iou_'+ str(th) + '_mtmc_' + info_label + '.xls'),index=False)
#     else:
#         summary.to_excel(os.path.join(results_path,  'iou_'+ str(th) + '_mtmc_' + info_label + '.xls'),index=False)
#     summary.to_csv(os.path.join(results_path,  'metrics_' + 'mtmc_' + info_label + '.csv'),index=False)
#
a=1