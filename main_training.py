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
import cv2
import pandas as pd
import torch
import argparse
import torch.nn as nn

from torch.utils.data import DataLoader

from datasets import dataset
from models.mpn import MOTMPNet


import utils
from sklearn.metrics.pairwise import paired_distances


from train import train, validate
from reid.extract_image_feat import ReidFeature
from libs import getConfiguration
import pathlib



def load_reid_model(CONFIG):
    reid_model = ReidFeature(0, CONFIG)
    #     model = ReidFeature(gpu_id.get(), _cfg)
    # reid_feat_numpy = model.extract(chunk_list[0])

    reid_model.model.cuda()
    if CONFIG['finetune']== True or CONFIG['finetune']== 'True':
        reid_model.model.train()
        ct = 0
        for child in reid_model.model.base.children():

            if ct < 7:
                # print(child)
                for param in child.parameters():
                    param.requires_grad = False


            ct = ct + 1
    else:
        reid_model.model.eval()


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
# USE_CUDA = torch.cuda.is_available()
# torch.cuda.device(1)


date = date = str(time.localtime().tm_year) + '-' + str(time.localtime().tm_mon).zfill(2) + '-' + str(
    time.localtime().tm_mday).zfill(2) + \
              ' ' + str(time.localtime().tm_hour).zfill(2) + ':' + str(time.localtime().tm_min).zfill(2) + ':' + str(
    time.localtime().tm_sec).zfill(2)


parser = argparse.ArgumentParser(description='Training GNN for MTMC tracking')
parser.add_argument('--Mode', metavar='DIR', help='Configuration file path')
parser.add_argument('--Options', metavar='DIR', nargs='+', help='an integer for the accumulator')


# Decode CONFIG file information
args = parser.parse_args()
CONFIG = getConfiguration.getConfiguration(args)
# CONFIG = yaml.safe_load(open(args.ConfigPath, 'r'))


if CONFIG['TRAINING']['only_dist'] or CONFIG['TRAINING']['only_appearance']:
    CONFIG['GRAPH_NET_PARAMS']['encoder_feats_dict']['edges']['edge_in_dim'] = 2
    CONFIG['GRAPH_NET_PARAMS']['encoder_feats_dict']['edges']['edge_out_dim'] = 4
    CONFIG['GRAPH_NET_PARAMS']['edge_model_feats_dict']['fc_dims'] = [4]
    CONFIG['GRAPH_NET_PARAMS']['classifier_feats_dict']['edge_in_dim'] = 4
    CONFIG['GRAPH_NET_PARAMS']['classifier_feats_dict']['edge_fc_dims'] = [2]

cnn_model_path  = pathlib.PurePath(CONFIG['CNN_MODEL']['reid_model'])
cnn_model_name = cnn_model_path.name[:-4]
if CONFIG['COMMENT'] is not None:
    if CONFIG['TRAINING']['loss_weight_custom'] == True or  CONFIG['TRAINING']['loss_weight_custom'] == 'True' :
        results_path = os.path.join(os.getcwd(), 'results',
                                    'tr_' + '-'.join(CONFIG['DATASET']['data_train']) + '_val_' +
                                    '-'.join(CONFIG['DATASET']['data_val']) + '_' + cnn_model_name + '_weight_custom_'+ str(CONFIG['TRAINING']['optimizer_type']) + '_lr_' + str(
                                        CONFIG['TRAINING']['lr'])
                                    + '_BS_' + str(CONFIG['TRAINING']['bs_train']) + '_' + str(
                                        CONFIG['TRAINING']['bs_val']) + '_' + str(CONFIG['COMMENT']) + '_' + date)
    else:

        results_path = os.path.join(os.getcwd(), 'results', 'tr_' + '-'.join(CONFIG['DATASET']['data_train']) + '_val_' +
                                '-'.join(CONFIG['DATASET']['data_val']) + '_'+ cnn_model_name + '_weight_' + str(CONFIG['TRAINING']['loss_weight'])
                                +'_' + str(CONFIG['TRAINING']['optimizer_type']) + '_lr_' + str(CONFIG['TRAINING']['lr'])
                                + '_BS_' + str(CONFIG['TRAINING']['bs_train'])+ '_' + str(CONFIG['TRAINING']['bs_val'])+ '_'  + str(CONFIG['COMMENT']) + '_'+ date)
else:

    if CONFIG['TRAINING']['loss_weight_custom'] == True or  CONFIG['TRAINING']['loss_weight_custom'] == 'True':

        results_path = os.path.join(os.getcwd(), 'results', 'tr_' + '-'.join(CONFIG['DATASET']['data_train']) + '_val_' +
                                    '-'.join(CONFIG['DATASET']['data_val']) + '_' + cnn_model_name + '_weight_custom_' + str(CONFIG['TRAINING']['optimizer_type']) + '_lr_' + str(CONFIG['TRAINING']['lr'])
                                    + '_BS_' + str(CONFIG['TRAINING']['bs_train']) + '_' + str(CONFIG['TRAINING']['bs_val']) + '_' + date)
    else:
        results_path = os.path.join(os.getcwd(), 'results',
                                    'tr_' + '-'.join(CONFIG['DATASET']['data_train']) + '_val_' +
                                    '-'.join(CONFIG['DATASET']['data_val']) + '_' + cnn_model_name + '_weight_' + str(
                                        CONFIG['TRAINING']['loss_weight'])
                                    + '_' + str(CONFIG['TRAINING']['optimizer_type']) + '_lr_' + str(
                                        CONFIG['TRAINING']['lr'])
                                    + '_BS_' + str(CONFIG['TRAINING']['bs_train']) + '_' + str(
                                        CONFIG['TRAINING']['bs_val']) + '_' + date)

os.mkdir(results_path)
os.mkdir(os.path.join(results_path, 'images'))
os.mkdir(os.path.join(results_path, 'models'))

with open(os.path.join(results_path, 'config_training.yaml'), 'w') as file:
    yaml.safe_dump(CONFIG, file,sort_keys = False)

shutil.copyfile('train.py', os.path.join(results_path, 'train.py'))
shutil.copyfile('main_training.py', os.path.join(results_path, 'main_training.py'))

cnn_model = load_reid_model(CONFIG['CNN_MODEL'])

##TRAIN DATASETS
train_datasets = []

for d in CONFIG['DATASET']['data_train']:

    train_datasets.append(dataset.AIC_dataset(d, 'train', CONFIG, cnn_model))

if len(train_datasets) > 1:
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    weights = list([])
    for t in train_datasets:
        weights.append(np.ones(len(t)) * (1 / (len(t))))
    weights = torch.from_numpy(np.asarray(np.concatenate(weights)))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),
                                                             replacement=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(CONFIG['TRAINING']['bs_train']),
                                               sampler=sampler,
                                               num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], collate_fn=my_collate,
                                               pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'], )


else:
    train_dataset = train_datasets[0]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(CONFIG['TRAINING']['bs_train']),
                                               shuffle=True,
                                               num_workers=CONFIG['DATALOADER']['NUM_WORKERS'],collate_fn=my_collate,
                                               pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])


## VALIDATION DATASETs

val_datasets = []

for d in CONFIG['DATASET']['data_val']:

    val_datasets.append(dataset.AIC_dataset(d, 'validation', CONFIG, cnn_model))


if len(val_datasets) > 1:
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    weights = list([])
    for t in val_datasets:
        weights.append(np.ones(len(t)) * (1 / (len(t))))
    weights = torch.from_numpy(np.asarray(np.concatenate(weights)))
    val_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),
                                                             replacement=False)

    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(CONFIG['TRAINING']['bs_val']),
                                                    shuffle=False, sampler=val_sampler,
                                                    num_workers=CONFIG['DATALOADER']['NUM_WORKERS'],
                                                    collate_fn=my_collate,
                                                    pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])

else:
    val_dataset = val_datasets[0]
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(CONFIG['TRAINING']['bs_val']),
                                                    shuffle=False,
                                                    num_workers=CONFIG['DATALOADER']['NUM_WORKERS'],
                                                    collate_fn=my_collate,
                                                    pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])



# LOAD MPN NETWORK#

mpn_model = load_model_mpn(CONFIG)
mpn_model.cuda()
num_params_mpn  = sum([np.prod(p.size()) for p in mpn_model.parameters()])

## LOSS AND OPTIMIZER

weights = []
if CONFIG['TRAINING']['optimizer_type']  == 'Adam':
    if CONFIG['TRAINING']['warmup_enable']:
        lr_warmup_list = np.linspace(CONFIG['TRAINING']['initial_lr'], CONFIG['TRAINING']['lr'],
                                     CONFIG['TRAINING']['warm_up_epochs'] + 1, endpoint=False)
        lr_warmup_list = lr_warmup_list[1:]

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mpn_model.parameters()),
                                    lr_warmup_list[0])
        flag_warmup_ended = False
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mpn_model.parameters()), lr=CONFIG['TRAINING']['OPTIMIZER']['args']['lr'])

elif CONFIG['TRAINING']['optimizer_type']  == 'SGD':
    if CONFIG['TRAINING']['warmup_enable']:
        lr_warmup_list = np.linspace(CONFIG['TRAINING']['initial_lr'], CONFIG['TRAINING']['lr'],
                                     CONFIG['TRAINING']['warm_up_epochs'] + 1, endpoint=False)
        lr_warmup_list = lr_warmup_list[1:]


        if CONFIG['CNN_MODEL']['finetune'] != True and  CONFIG['CNN_MODEL']['finetune'] != 'True':
            optimizer = torch.optim.SGD(mpn_model.parameters(),
                                        lr= lr_warmup_list[0],
                                        momentum=CONFIG['TRAINING']['momentum'],
                                        weight_decay=CONFIG['TRAINING']['weight_decay'])
        else:

            params = list(mpn_model.parameters()) + list(filter(lambda p: p.requires_grad, cnn_model.model.parameters()))
            optimizer = torch.optim.SGD(params,
                                        lr=lr_warmup_list[0],
                                        momentum=CONFIG['TRAINING']['momentum'],
                                        weight_decay=CONFIG['TRAINING']['weight_decay'])

        flag_warmup_ended = False
    else:
        if CONFIG['CNN_MODEL']['finetune'] != True and CONFIG['CNN_MODEL']['finetune'] != 'True':
            optimizer = torch.optim.SGD(mpn_model.parameters(),
                                        lr =CONFIG['TRAINING']['lr'],
                                        momentum=CONFIG['TRAINING']['momentum'],
                                        weight_decay=CONFIG['TRAINING']['weight_decay'])
        else:

            params = list(mpn_model.parameters()) + list(filter(lambda p: p.requires_grad, cnn_model.model.parameters()))
            optimizer = torch.optim.SGD(params,
                                        lr = CONFIG['TRAINING']['lr'],
                                        momentum = CONFIG['TRAINING']['momentum'],
                                        weight_decay = CONFIG['TRAINING']['weight_decay'])


    # Learning rate decay
        if CONFIG['TRAINING']['scheduler_type'] == 'STEP':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG['TRAINING']['step_size'],
                                                        gamma=CONFIG['TRAINING']['gamma'])
        elif CONFIG['TRAINING']['scheduler_type'] == 'COSINE':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['TRAINING']['epochs'])
# TRAINING CRITERIONS
if CONFIG['TRAINING']['loss_name'] == 'Focal':
    alfa = torch.tensor([0.95])
    criterion = utils.FocalLoss(reduction= 'mean', alpha=alfa)
    criterion_no_reduction = utils.FocalLoss(reduction = 'none', alpha=alfa)
    weights = torch.tensor([])


elif CONFIG['TRAINING']['loss_name'] == 'BCE_weighted':
    criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(CONFIG['TRAINING']['loss_weight']))
    criterion_no_reduction = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(CONFIG['TRAINING']['loss_weight']))

elif CONFIG['TRAINING']['loss_name'] == 'BCE':
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    criterion_no_reduction = nn.BCEWithLogitsLoss(reduction='none')

elif  CONFIG['TRAINING']['loss_name'] == 'CE_weighted':
    if CONFIG['GRAPH_NET_PARAMS']['classifier_feats_dict']['edge_out_dim'] > 1:
        #USING Cross entropy instead of BCE

        if str(CONFIG['TRAINING']['loss_weight_custom']) == 'True':
            # Weight class is computed based on the unbalance of each batch
            # pos_weight = int(CONFIG['TRAINING']['loss_weight'])
            # weights = torch.tensor([1., pos_weight]).cuda()
            criterion = nn.CrossEntropyLoss(reduction='none')
            criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
            weights = torch.tensor([])
        else:
            # Weight class is fixed
            pos_weight = int(CONFIG['TRAINING']['loss_weight'])
            weights = torch.tensor([1., pos_weight]).cuda()
            criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
            criterion_no_reduction = nn.CrossEntropyLoss(weight=weights, reduction='none')


# avg per epoch
training_loss_avg = []
training_loss_avg_1 = []
training_loss_avg_0 = []

training_precision_1_avg = []
training_precision_0_avg = []
training_FPR_avg = []

val_loss_avg = []
val_loss_avg_1 = []
val_loss_avg_0 = []

val_precision_1_avg = []
val_precision_0_avg = []
val_prec_in_history = []

val_FPR_avg = []


# all, per iteration
train_loss_in_history = []
train_prec0_in_history = []
train_prec1_in_history = []
train_prec_in_history = []

val_loss_in_history = []
val_prec0_in_history = []
val_prec1_in_history = []


## TRAINING
best_prec = 0
best_val_loss = 1000
list_lr = list([])


nsteps = CONFIG['GRAPH_NET_PARAMS']['num_class_steps']
list_mean_probs_history = {"0": {}, "1": {}}

if nsteps > 0:
    for i in range(nsteps):
        list_mean_probs_history["0"]["step" + str(i)] = []
        list_mean_probs_history["1"]["step" + str(i)] = []

    list_mean_probs_history_val = {"0": {}, "1":{}}
    for i in range(nsteps):
        list_mean_probs_history_val["0"]["step" + str(i)] = []
        list_mean_probs_history_val["1"]["step" + str(i)] = []

print('Starting training')
for epoch in range(0, CONFIG['TRAINING']['epochs']):
    epoch_start = time.time()
    list_lr.append(optimizer.param_groups[0]['lr'])


    train_losses,train_losses1, train_losses0, train_precision_1, train_precision_0,train_loss_in_history,\
    train_prec1_in_history,train_prec0_in_history,train_prec_in_history,list_mean_probs_history, train_FPR  = \
        train(CONFIG, train_loader, cnn_model, mpn_model, epoch, optimizer,results_path,train_loss_in_history, \
              train_prec1_in_history,train_prec0_in_history,train_prec_in_history, criterion, criterion_no_reduction,list_mean_probs_history, weights)

    training_loss_avg.append(train_losses.avg)
    training_loss_avg_1.append(train_losses1.avg)
    training_loss_avg_0.append(train_losses0.avg)

    training_precision_1_avg.append(train_precision_1.avg)
    training_precision_0_avg.append(train_precision_0.avg)
    training_FPR_avg.append(train_FPR.avg)

    val_losses, val_losses1, val_losses0, val_precision_1, val_precision_0,val_loss_in_history,val_prec1_in_history,val_prec0_in_history,val_prec_in_history,list_mean_probs_history_val, val_FPR = \
        validate(CONFIG,validation_loader, cnn_model, mpn_model, results_path,epoch,val_loss_in_history,val_prec1_in_history,val_prec0_in_history,val_prec_in_history,
                 list_mean_probs_history_val ,criterion, criterion_no_reduction)

    val_loss_avg.append(val_losses.avg)
    val_loss_avg_1.append(val_losses1.avg)
    val_loss_avg_0.append(val_losses0.avg)

    val_precision_1_avg.append(val_precision_1.avg)
    val_precision_0_avg.append(val_precision_0.avg)

    val_FPR_avg.append(val_FPR.avg)

    if CONFIG['TRAINING']['warmup_enable'] and not (flag_warmup_ended):
        if epoch == CONFIG['TRAINING']['warm_up_epochs']:
            flag_warmup_ended = True

        if (flag_warmup_ended):
            # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mpn_model.parameters()), lr=CONFIG['TRAINING']['OPTIMIZER']['args']['lr'])
            optimizer = torch.optim.SGD(mpn_model.parameters(),
                                        lr=CONFIG['TRAINING']['lr'],
                                        momentum=CONFIG['TRAINING']['momentum'],
                                        weight_decay=CONFIG['TRAINING']['weight_decay'])
            # Learning rate decay SGD
            if CONFIG['TRAINING']['scheduler_type'] == 'STEP':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG['TRAINING']['step_size'], gamma=CONFIG['TRAINING']['gamma'])
            elif CONFIG['TRAINING']['scheduler_type'] == 'COSINE':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['TRAINING']['epochs'])

        else:
            optimizer.param_groups[0]['lr'] = lr_warmup_list[epoch]

    else:
        if CONFIG['TRAINING']['optimizer_type'] == 'SGD':
            scheduler.step()


    utils.save_graphics(results_path,training_precision_1_avg,training_precision_0_avg,val_precision_1_avg,val_precision_0_avg,training_loss_avg, training_loss_avg_1,
                        training_loss_avg_0,val_loss_avg,val_loss_avg_1,val_loss_avg_0,list_lr, training_FPR_avg, val_FPR_avg )



    # is_best = (val_precision_1.avg + val_precision_0.avg)/2 > best_prec
    is_best = (val_loss_avg[-1]) < best_val_loss
    # best_prec = max((val_precision_1.avg + val_precision_0.avg)/2, best_prec)
    best_val_loss = min(val_loss_avg[-1],best_val_loss)

    # SAVE

    utils.save_checkpoint({
        'epoch': epoch + 1,
        'model_state_dict': mpn_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'prec': (val_precision_1.avg + val_precision_0.avg)/2,
        'prec1': val_precision_1_avg,
        'prec0': val_precision_0_avg,
        'best_loss':  best_val_loss,
        'model_parameters': num_params_mpn,

        'CONFIG': CONFIG
    }, is_best, results_path)


    print('Elapsed time for epoch {}: {time:.3f} minutes'.format(epoch, time=(time.time() - epoch_start) / 60))

