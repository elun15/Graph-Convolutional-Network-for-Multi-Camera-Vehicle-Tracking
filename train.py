
import os
import time
import shutil
import yaml
import datetime

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from torch_geometric.utils import to_networkx

import utils


list_cam_colors = list(['royalblue', 'darkorange','green','firebrick'])


#####################
# Construct Weights #
#####################
#
# n_0 = len(labels_edges_GT)-sum(labels_edges_GT)
# n_1 = sum(labels_edges_GT)
# w_0 = (n_0 + n_1) / (2.0 * n_0)
# w_1 = (n_0 + n_1) / (2.0 * n_1)


def compute_loss_acc(outputs, batch, criterion, criterion_no_reduction,  mode, flag_BCE,FPR_flag, FPR_alpha,
                     loss_weight_custom, loss_name,  weights=None):

    # Define Balancing weight

    labels = batch.edge_labels.view(-1)

    # Compute Weighted BCE:
    loss = 0
    loss2 = 0
    loss_class1 = 0
    loss_class0 = 0
    precision_class1 = list()
    precision_class0 = list()
    precision_all = list()

    list_pred_prob = list()
    num_steps = len(outputs['classified_edges'])

   # Compute loss of all the steps and sum them

    ## FOR CONSIDERING ONLY LAST 3 STEPS or less
    # step_ini = max(0,num_steps-3)
    # step_end = num_steps

    # comment FOR CONSIDERING ALL STEPS
    step_ini= 0 #0
    step_end = num_steps

    for step in range(step_ini, step_end):


        # FOR BinaryCE and BinaryCEweighed
        if flag_BCE:
            preds = outputs['classified_edges'][step].view(-1)
            if mode == 'train':

                loss_per_sample = criterion_no_reduction(preds, labels)
                loss += criterion(preds, labels) # before +=

                loss_class1 += torch.mean(loss_per_sample[labels == 1])
                loss_class0 += torch.mean(loss_per_sample[labels == 0])


            else:
                loss_per_sample = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
                # loss_per_sample = criterion_no_reduction(preds, labels)

                loss_class1 += torch.mean(loss_per_sample[labels == 1])
                loss_class0 += torch.mean(loss_per_sample[labels == 0])

                loss += F.binary_cross_entropy_with_logits(preds, labels, reduction='mean')
                # loss += criterion(preds, labels) # before +=

            with torch.no_grad():
                sig = torch.nn.Sigmoid()
                preds_prob = sig(preds)
                list_pred_prob.append(preds_prob.cpu())

        # FOR CEw
        else:
            preds = outputs['classified_edges'][step]
            labels = labels.long()
            predictions = torch.argmax(preds, dim=1)
            FP = torch.sum(predictions[labels == 0])
            TN = (predictions[labels == 0]).shape[0] - FP
            FPR = FP / (FP + TN)

            TP = torch.sum(predictions[labels == 1])
            TPR = TP / (TP + TN)


            if mode == 'train':
                if loss_name != 'Focal':
                    if loss_weight_custom == 'False' or loss_weight_custom == False:
                    # if len(weights) > 0:
                        # weights are fixed and given by parameters

                        loss_per_sample = criterion_no_reduction(preds, labels)
                        loss_per_sample = loss_per_sample / weights[labels].sum()
                        # loss_total = loss_per_sample.sum() / weights[labels].sum()
                        loss_class1 += torch.sum(loss_per_sample[labels == 1])
                        loss_class0 += torch.sum(loss_per_sample[labels == 0])

                        loss += criterion(preds, labels)

                    else:
                    # weight is computed online per batch
                        loss_per_sample = criterion_no_reduction(preds, labels)
                        n_0 = len(labels)-sum(labels)
                        n_1 = sum(labels)
                        w_0b = (n_0 + n_1) / (n_0) #TODO quitar el 2
                        w_1b = (n_0 + n_1) / (n_1)
                        w_1 = w_1b / w_0b
                        w_0 = w_0b / w_0b
                        custom_weights = torch.tensor([w_0, w_1])

                        # chequear
                        loss_per_sample[labels == 0] = (loss_per_sample[labels == 0] * w_0)
                        loss_per_sample[labels == 1] = (loss_per_sample[labels == 1] * w_1)
                        loss_per_sample = loss_per_sample / custom_weights[labels].sum()

                        # FP_idx_samples = np.intersect1d(torch.where(labels == 0)[0].cpu(), torch.where(predictions == 1)[0].cpu())
                        # loss_per_sample[ FP_idx_samples] = loss_per_sample[FP_idx_samples] * 2
                        loss_class1 += torch.sum(loss_per_sample[labels == 1])
                        loss_class0 += torch.sum(loss_per_sample[labels == 0])
                        loss += torch.sum(loss_per_sample)

                else:
        #         If Focal loss: REVISAR CODIGOS  ERROR DE CUDA; comprobar que gammana = 0 da igual que con la CE

                    if loss_weight_custom == True or loss_weight_custom == 'True':
                        n_0 = len(labels) - sum(labels)
                        n_1 = sum(labels)
                        w_0b = (n_0 + n_1) / ( 2* n_0)
                        w_1b = (n_0 + n_1) / (2 * n_1)
                        w_1 = w_1b / w_0b
                        w_0 = w_0b / w_0b

                        w_1_f = w_1/w_1
                        w_0_f = w_0/w_1

                        # alfa = torch.tensor([1-w_0_f]).cuda()
                        alfa = 1-w_0_f
                        criterion_no_reduction = utils.FocalLoss(reduction='none', alpha=alfa)

                        # criterion = utils.FocalLoss(gamma = 0, reduction='sum', alpha=torch.tensor([1-alfa, 1]).cuda())
                        # criterion_no_reduction = utils.FocalLoss(gamma = 0,reduction='none', alpha=torch.tensor([1, 1-alfa]).cuda())
                        # input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
                        # loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy

                        loss_per_sample = criterion_no_reduction(preds, labels)
                        # loss_total = loss_per_sample.sum() / weights[labels].sum()
                        loss_class1 += torch.mean(loss_per_sample[labels == 1])
                        loss_class0 += torch.mean(loss_per_sample[labels == 0])

                        loss += criterion(preds, labels)

                    else:
                        loss_per_sample = criterion_no_reduction(preds, labels)

                        # loss_total = loss_per_sample.sum() / weights[labels].sum()
                        loss_class1 += torch.mean(loss_per_sample[labels == 1])
                        loss_class0 += torch.mean(loss_per_sample[labels == 0])

                        loss += criterion(preds, labels)


            else:
                loss_per_sample = F.cross_entropy(preds, labels, reduction='none')
                # loss_per_sample = loss_per_sample.sum() / weights[labels].sum()

                loss_class1 += torch.mean(loss_per_sample[labels == 1])
                loss_class0 += torch.mean(loss_per_sample[labels == 0])
                # loss += torch.sum(loss_per_sample)

                loss += F.cross_entropy(preds, labels, reduction='mean')

            if FPR_flag:
                loss += float(FPR_alpha) * FPR
                # loss += 1 * (1-TPR)

            with torch.no_grad():
                sof = torch.nn.Softmax(dim=1)
                preds_prob = sof(preds)[:, 1]
                list_pred_prob.append(preds_prob.cpu())


    # Precision is computed only with last step predictions

    with torch.no_grad():

        if flag_BCE:
            preds = outputs['classified_edges'][-1].view(-1)
            sig = torch.nn.Sigmoid()
            preds_prob = sig(preds)
            predictions = (preds_prob >= 0.5) * 1
        else:
            preds = outputs['classified_edges'][-1]
            sof = torch.nn.Softmax(dim=1)
            preds_prob = sof(preds)

            predictions = torch.argmax(preds, dim=1)


        # Precision class 1
        index_label_1 = np.where(np.asarray(labels.cpu()) == 1)
        sum_successes_1 = np.sum(predictions.cpu().numpy()[index_label_1] == labels.cpu().numpy()[index_label_1])
        if sum_successes_1 == 0:
            precision_class1.append(0)
        else:
            precision_class1.append((sum_successes_1 / len(labels[index_label_1])) * 100.0)

        # Precision class 0
        index_label_0 = np.where(np.asarray(labels.cpu()) == 0)
        sum_successes_0 = np.sum(predictions.cpu().numpy()[index_label_0] == labels.cpu().numpy()[index_label_0])
        if sum_successes_0 == 0:
            precision_class0.append(0)
        else:
            precision_class0.append((sum_successes_0 / len(labels[index_label_0])) * 100.0)

        # Precision
        sum_successes = np.sum(predictions.cpu().numpy() == labels.cpu().numpy())
        if sum_successes == 0:
            precision_all.append(0)
        else:
            precision_all.append((sum_successes / len(labels) )* 100.0)


    return loss, precision_class1, precision_class0, precision_all, loss_class1, loss_class0, list_pred_prob, FPR


def train(CONFIG, train_loader, reid_model, mpn_model, epoch, optimizer,results_path,train_loss_in_history,
          train_prec1_in_history,train_prec0_in_history, train_prec_in_history, criterion, criterion_no_reduction,list_mean_probs_history, weights=None):

    train_losses = utils.AverageMeter('losses', ':.4e')
    train_losses1 = utils.AverageMeter('losses', ':.4e')
    train_losses0 = utils.AverageMeter('losses', ':.4e')
    train_FPR = utils.AverageMeter('FPR','d' )


    train_batch_time = utils.AverageMeter('batch_time', ':6.3f')
    train_precision_class1 = utils.AverageMeter('Precision_class1', ':6.2f')
    train_precision_class0 = utils.AverageMeter('Precision_class0', ':6.2f')
    train_precision = utils.AverageMeter('Precision', ':6.2f')
    mpn_model.train()

    n_steps = CONFIG['GRAPH_NET_PARAMS']['num_class_steps']
    list_mean_probs = {"0": {}, "1": {}}
    if n_steps >0:
        for i in range(n_steps):
            list_mean_probs["0"]["step" + str(i)] = []
            list_mean_probs["1"]["step" + str(i)] = []

    else:
        n_steps = 1
        for i in range(n_steps):
            list_mean_probs["0"]["step" + str(i)] = []
            list_mean_probs["1"]["step" + str(i)] = []


    for i, data in enumerate(train_loader):

        if i >= 0 :

            start_time = time.time()

            ########### Data extraction ###########
            bboxes = data
            #Structure of bboxes:
            #   len(bboxes) = batch size
            #   bboxes[X] = {'car ID': {'cam ID': tensor of stacked bboxes of the track }}

            num_ids = len(bboxes)
            node_embeds_g = []
            node_labels_g = []
            cam_ids_nodes = []

            # COMPUTING NODE EMBEDDINGS,
            for n in range(num_ids):
                id = bboxes[n]['id']
                cams = bboxes[n]['cam']

                for c in range(len(cams)):
                    imgs_bboxes = bboxes[n]['bboxes'][c]
                    node_labels_g.append(id)
                    cam_ids_nodes.append(bboxes[n]['cam'][c])

                    if CONFIG['CNN_MODEL']['finetune'] == True or CONFIG['CNN_MODEL']['finetune'] == 'True':
                        bboxes_embeds = reid_model.model(imgs_bboxes.cuda())
                        node_embed = torch.mean(bboxes_embeds, 0)
                        node_embeds_g.append(node_embed)
                    else:
                        with torch.no_grad():
                            bboxes_embeds = reid_model.model(imgs_bboxes.cuda())
                            node_embed = torch.mean(bboxes_embeds,0)
                            node_embeds_g.append(node_embed)
                            #Normalizadas?  node_embeds =  F.normalize(node_embeds, p= 2,dim=0)


            node_embeds_g = torch.stack(node_embeds_g)

            if CONFIG['CNN_MODEL']['L2norm']:
                node_embeds_g = F.normalize(node_embeds_g, p=2, dim=0)

            # COMPUTING EDGE INFORMATION
            a=1
            edge_ixs_g = []
            nodes = np.asarray(range(len(cam_ids_nodes)))
            for id_cam in np.unique(cam_ids_nodes):
                ids_in_cam = nodes[np.asarray(cam_ids_nodes) == id_cam]
                ids_out_cam = nodes[np.asarray(cam_ids_nodes) != id_cam]
                edge_ixs_g.append(torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))
            edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
            edge_ixs_g_np = edge_ixs_g.cpu().numpy()

            # # EDGE LABELS
            node_labels_g_np = np.asarray(node_labels_g)
            edge_labels_g = torch.from_numpy(np.asarray([1 if (node_labels_g_np[nodes == edge_ixs_g_np[0][i]] ==
                                  node_labels_g_np[nodes == edge_ixs_g_np[1][i]]) else 0
                            for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()


            # EDGE EMBEDDING
            node_dist_g = F.pairwise_distance(node_embeds_g[edge_ixs_g[0]], node_embeds_g[edge_ixs_g[1]]).view(-1, 1)
            node_dist_g_cos = 1-F.cosine_similarity(node_embeds_g[edge_ixs_g[0]], node_embeds_g[edge_ixs_g[1]]).view(-1, 1)
            edge_attr = torch.cat((node_dist_g, node_dist_g_cos), dim=1)

            data = Data(x=node_embeds_g, edge_index=edge_ixs_g, y=node_labels_g, edge_attr=edge_attr, edge_labels=edge_labels_g)

            if CONFIG['VISUALIZE']:
                edge_labels_np = np.asarray(edge_labels_g.cpu())
                node_labels_g_np = np.asarray(node_labels_g)

                G = to_networkx(data, to_undirected=True)
                utils.visualize(G, color=data.y, edge_labels=edge_labels_np, edge_index=edge_ixs_g_np,
                                node_label=node_labels_g_np, mode = 'train')
            #
            ########### Forward ###########

            outputs, latent_node_features = mpn_model(data)

            ########### Loss ###########
            if CONFIG['GRAPH_NET_PARAMS']['classifier_feats_dict']['edge_out_dim'] == 1:
                flag_BCE = True
            else:
                flag_BCE = False

            if CONFIG['TRAINING']['add_FPR'] == True or CONFIG['TRAINING']['add_FPR'] == 'True':
                FPR_flag = True
            else:
                FPR_flag = False

            loss, precision1, precision0, precision,loss_class1, loss_class0, list_pred_probs, FPR = compute_loss_acc(outputs, data, criterion, criterion_no_reduction, mode='train', flag_BCE=flag_BCE, FPR_flag=FPR_flag,
                                                                                                                                   FPR_alpha=CONFIG['TRAINING']['FPR_alpha'], loss_weight_custom=CONFIG['TRAINING']['loss_weight_custom'],
                                                                                                                      loss_name=CONFIG['TRAINING']['loss_name'],weights=weights)
            # loss = loss * 100
            #Fill dictionary with mean probabilities of each class at each step
            if flag_BCE:
                if len(list_pred_probs) > 0:
                    nsteps = len(list_pred_probs)
                    for s in range(nsteps):
                        if sum(sum([data.edge_labels == 0])) == 0:
                            list_mean_probs["0"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                        else:
                            list_mean_probs["0"]["step" + str(s)].append(torch.mean(list_pred_probs[s][data.edge_labels == 0]))
                        if sum(sum([data.edge_labels == 1])) == 0:
                            list_mean_probs["1"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                        else:
                            list_mean_probs["1"]["step" + str(s)].append(torch.mean(list_pred_probs[s][data.edge_labels == 1]))

            else:
                #list_pred_probs contains the probabilities of class 1, class 0 is 1-x
                if len(list_pred_probs) > 0:
                    nsteps = len(list_pred_probs)
                    for s in range(nsteps):
                        if sum(sum([data.edge_labels == 0])) == 0:
                            list_mean_probs["0"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                        else:
                            list_mean_probs["0"]["step" + str(s)].append(torch.mean(1-(list_pred_probs[s][data.edge_labels == 0])))
                        if sum(sum([data.edge_labels == 1])) == 0:
                            list_mean_probs["1"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                        else:
                            list_mean_probs["1"]["step" + str(s)].append(torch.mean(list_pred_probs[s][data.edge_labels == 1]))


            train_losses.update(loss.item(), int(CONFIG['TRAINING']['bs_train']))
            train_losses1.update(loss_class1.item(), int(CONFIG['TRAINING']['bs_train']))
            train_losses0.update(loss_class0.item(), int(CONFIG['TRAINING']['bs_train']))


            train_precision_class1.update(np.sum(np.asarray([item for item in precision1])) / len(precision1),int(CONFIG['TRAINING']['bs_train'] ))
            train_precision_class0.update(np.sum(np.asarray([item for item in precision0])) / len(precision0),int(CONFIG['TRAINING']['bs_train'] ))
            train_precision.update(np.sum(np.asarray([item for item in precision])) / len(precision),int(CONFIG['TRAINING']['bs_train'] ))

            train_FPR.update(FPR.item(),int(CONFIG['TRAINING']['bs_train']))

            # accuracies.append()
            train_loss_in_history.append(train_losses.avg)
            train_prec1_in_history.append(np.sum(np.asarray([item for item in precision1])) / len(precision1))
            train_prec0_in_history.append(np.sum(np.asarray([item for item in precision0])) / len(precision0))
            train_prec_in_history.append(np.sum(np.asarray([item for item in precision])) / len(precision))

            ########### Accuracy ###########

            ########### Optimizer update ###########

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(lambda: float(loss))

            train_batch_time.update(time.time() - start_time)

            if i % 1  == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                      'Train Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                      'Train Acc 1 {acc.val:.3f} (avg: {acc.avg:.3f})\t'
                      'Train Acc 0 {acc2.val:.3f} (avg: {acc2.avg:.3f})\t'
                      '{et}<{eta}'.format(epoch, i, len(train_loader), batch_time=train_batch_time,  loss=train_losses,
                                          acc = train_precision_class1, acc2 =train_precision_class0, et=str(datetime.timedelta(seconds=int(train_batch_time.sum))),
                                          eta=str(datetime.timedelta(seconds=int(train_batch_time.avg * (len(train_loader) - i))))))

    if len(list_mean_probs["0"]["step0"]) > 0:
        plt.figure(1)
        for i in range(nsteps):
            list_mean_probs_history["0"]["step" + str(i)].append(
                np.mean(torch.stack(list_mean_probs["0"]["step" + str(i)]).cpu().numpy()))
            plt.plot(list_mean_probs_history["0"]["step" + str(i)], '--', label="Class 0 Iter" + str(i))
            list_mean_probs_history["1"]["step" + str(i)].append(
                np.mean(torch.stack(list_mean_probs["1"]["step" + str(i)]).cpu().numpy()))
            plt.plot(list_mean_probs_history["1"]["step" + str(i)], '-', label="Class 1 Iter" + str(i))
        plt.legend(loc='best')
        plt.savefig(results_path + '/images/Mean Probability per Class per Epoch Training.pdf', bbox_inches='tight')
        plt.close()

    plt.figure(1)
    plt.plot(train_loss_in_history, label='Loss')

    plt.ylabel('Loss'), plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Training Loss per Iteration.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(1)
    plt.plot(train_prec1_in_history,'g', label='Precision class 1')
    plt.plot(train_prec0_in_history, 'r', label='Precission class 0')

    plt.ylabel('Precision'), plt.xlabel('Iteration')
    plt.legend(loc= 'best')
    plt.savefig(results_path + '/images/Training Precision per Iteration.pdf', bbox_inches='tight')
    plt.close()

    return train_losses, train_losses1, train_losses0, train_precision_class1, train_precision_class0, train_loss_in_history,train_prec1_in_history,train_prec0_in_history,train_prec_in_history,list_mean_probs_history, train_FPR


def validate(CONFIG, val_loader, reid_model, mpn_model, results_path,epoch,val_loss_in_history,val_prec1_in_history,val_prec0_in_history,val_prec_in_history,list_mean_probs_history_val,criterion, criterion_no_reduction):
    val_losses = utils.AverageMeter('losses', ':.4e')
    val_losses1 = utils.AverageMeter('losses', ':.4e')
    val_losses0 = utils.AverageMeter('losses', ':.4e')

    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')
    val_precision_1 = utils.AverageMeter('Val prec class 1', ':6.2f')
    val_precision_0 = utils.AverageMeter('Val prec class 0', ':6.2f')
    val_precision = utils.AverageMeter('Val prec', ':6.2f')
    val_FPR = utils.AverageMeter('FPR', 'd')

    mpn_model.eval()

    n_steps = CONFIG['GRAPH_NET_PARAMS']['num_class_steps']
    list_mean_probs = {"0": {}, "1": {}}
    # for i in range(nsteps):
    #     list_mean_probs["0"]["step" + str(i)] = []
    #     list_mean_probs["1"]["step" + str(i)] = []
    if n_steps >0:
        for i in range(n_steps):
            list_mean_probs["0"]["step" + str(i)] = []
            list_mean_probs["1"]["step" + str(i)] = []

    else:
        n_steps = 1
        for i in range(n_steps):
            list_mean_probs["0"]["step" + str(i)] = []
            list_mean_probs["1"]["step" + str(i)] = []


    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 0:

                start_time = time.time()

                ########### Data extraction ###########
                bboxes = data
                # Structure of bboxes:
                #   len(bboxes) = batch size
                #   bboxes[X] = {'car ID': {'cam ID': tensor of stacked bboxes of the track }}

                num_ids = len(bboxes)
                node_embeds_g = []
                node_labels_g = []
                cam_ids_nodes = []

                # COMPUTING NODE EMBEDDINGS,
                for n in range(num_ids):
                    id = bboxes[n]['id']
                    cams = bboxes[n]['cam']

                    for c in range(len(cams)):
                        imgs_bboxes = bboxes[n]['bboxes'][c]
                        node_labels_g.append(id)
                        cam_ids_nodes.append(bboxes[n]['cam'][c])

                        with torch.no_grad():
                            bboxes_embeds = reid_model.model(imgs_bboxes.cuda())
                            node_embed = torch.mean(bboxes_embeds, 0)
                            node_embeds_g.append(node_embed)

                node_embeds_g = torch.stack(node_embeds_g)
                if CONFIG['CNN_MODEL']['L2norm']:
                    node_embeds_g = F.normalize(node_embeds_g, p=2, dim=0)
                # COMPUTING EDGE INFORMATION
                a = 1
                edge_ixs_g = []
                nodes = np.asarray(range(len(cam_ids_nodes)))
                for id_cam in np.unique(cam_ids_nodes):
                    ids_in_cam = nodes[np.asarray(cam_ids_nodes) == id_cam]
                    ids_out_cam = nodes[np.asarray(cam_ids_nodes) != id_cam]
                    edge_ixs_g.append(torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))
                edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                edge_ixs_g_np = edge_ixs_g.cpu().numpy()

                # # EDGE LABELS
                node_labels_g_np = np.asarray(node_labels_g)
                edge_labels_g = torch.from_numpy(np.asarray([1 if (node_labels_g_np[nodes == edge_ixs_g_np[0][i]] ==
                                                                   node_labels_g_np[
                                                                       nodes == edge_ixs_g_np[1][i]]) else 0
                                                             for i in range(edge_ixs_g_np.shape[1])])).type(
                    torch.float).cuda()

                # EDGE EMBEDDING
                node_dist_g = F.pairwise_distance(node_embeds_g[edge_ixs_g[0]], node_embeds_g[edge_ixs_g[1]]).view(-1,
                                                                                                                   1)
                node_dist_g_cos = 1-F.cosine_similarity(node_embeds_g[edge_ixs_g[0]], node_embeds_g[edge_ixs_g[1]]).view(
                    -1, 1)
                edge_attr = torch.cat((node_dist_g, node_dist_g_cos), dim=1)

                data = Data(x=node_embeds_g, edge_index=edge_ixs_g, y=node_labels_g, edge_attr=edge_attr,
                            edge_labels=edge_labels_g)

                if CONFIG['VISUALIZE']:
                    edge_labels_np = np.asarray(edge_labels_g.cpu())
                    node_labels_g_np = np.asarray(node_labels_g)

                    G = to_networkx(data, to_undirected=True)
                    utils.visualize(G, color=data.y, edge_labels=edge_labels_np, edge_index=edge_ixs_g_np,
                                    node_label=node_labels_g_np, mode = 'val')

                ########### Forward ###########

                outputs, latent_node_features = mpn_model(data)


                ########### Loss ###########
                if CONFIG['GRAPH_NET_PARAMS']['classifier_feats_dict']['edge_out_dim'] == 1:
                    flag_BCE = True
                else:
                    flag_BCE = False

                if CONFIG['TRAINING']['add_FPR'] == True:
                    FPR_flag = True
                else:
                    FPR_flag = False
                # loss, acc_actives, acc_nonactives = compute_loss_acc(outputs, data_batch, mode = 'validate')
                loss, precision1, precision0, precision,loss_class1, loss_class0,list_pred_probs,FPR = compute_loss_acc(outputs, data, criterion =criterion, criterion_no_reduction=criterion_no_reduction,
                                                                                                                        mode='validate',flag_BCE=flag_BCE,FPR_flag=FPR_flag, loss_weight_custom=CONFIG['TRAINING']['loss_weight_custom'], loss_name=CONFIG['TRAINING']['loss_name'],
                                                                                                                        FPR_alpha = CONFIG['TRAINING']['FPR_alpha'])

                # Fill dictionary with mean probabilities of each class at each step
                if flag_BCE:
                    if len(list_pred_probs) > 0:
                        nsteps = len(list_pred_probs)
                        for s in range(nsteps):
                            if sum(sum([data.edge_labels == 0])) == 0:
                                list_mean_probs["0"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                            else:
                                list_mean_probs["0"]["step" + str(s)].append(
                                    torch.mean(list_pred_probs[s][data.edge_labels == 0]))
                            if sum(sum([data.edge_labels == 1])) == 0:
                                list_mean_probs["1"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                            else:
                                list_mean_probs["1"]["step" + str(s)].append(
                                    torch.mean(list_pred_probs[s][data.edge_labels == 1]))

                else:
                    # list_pred_probs contains the probabilities of class 1, class 0 is 1-x
                    if len(list_pred_probs) > 0:
                        nsteps = len(list_pred_probs)
                        for s in range(nsteps):
                            if sum(sum([data.edge_labels == 0])) == 0:
                                list_mean_probs["0"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                            else:
                                list_mean_probs["0"]["step" + str(s)].append(
                                    torch.mean(1 - (list_pred_probs[s][data.edge_labels == 0])))
                            if sum(sum([data.edge_labels == 1])) == 0:
                                list_mean_probs["1"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                            else:
                                list_mean_probs["1"]["step" + str(s)].append(
                                    torch.mean(list_pred_probs[s][data.edge_labels == 1]))

                val_losses.update(loss.item(), int(CONFIG['TRAINING']['bs_val']))
                val_losses1.update(loss_class1.item(),  int(CONFIG['TRAINING']['bs_val']))
                val_losses0.update(loss_class0.item(),  int(CONFIG['TRAINING']['bs_val']))


                val_precision_1.update(np.sum(np.asarray([item for item in precision1])) / len(precision1),      int(CONFIG['TRAINING']['bs_val']))
                val_precision_0.update(np.sum(np.asarray([item for item in precision0])) / len(precision0),  int(CONFIG['TRAINING']['bs_val']))
                val_precision.update(np.sum(np.asarray([item for item in precision])) / len(precision),   int(CONFIG['TRAINING']['bs_val']))

                val_FPR.update(FPR.item(), int(CONFIG['TRAINING']['bs_train']))

                # accuracies.append()
                val_loss_in_history.append(val_losses.avg)
                val_prec1_in_history.append(np.sum(np.asarray([item for item in precision1])) / len(precision1))
                val_prec0_in_history.append(np.sum(np.asarray([item for item in precision0])) / len(precision0))
                val_prec_in_history.append(np.sum(np.asarray([item for item in precision])) / len(precision))

                ########### Accuracy ###########

                ########### Optimizer update ###########



                val_batch_time.update(time.time() - start_time)

                if i % 1 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                          'Val Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                          'Val Precision class 1 {acc.val:.3f} (avg: {acc.avg:.3f})\t'
                          'Val Precision class 0 {acc2.val:.3f} (avg: {acc2.avg:.3f})\t'
                          '{et}<{eta}'.format(i, len(val_loader), batch_time=val_batch_time, loss=val_losses,
                                              acc=val_precision_1, acc2=val_precision_0,
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(
                                                  seconds=int(val_batch_time.avg * (len(val_loader) - i))))))

    plt.figure(1)
    for i in range(nsteps):
        list_mean_probs_history_val["0"]["step" + str(i)].append(np.mean(torch.stack(list_mean_probs["0"]["step" + str(i)]).cpu().numpy()))
        plt.plot(list_mean_probs_history_val["0"]["step" + str(i)], '--', label="Class 0 Iter" + str(i))

        list_mean_probs_history_val["1"]["step" + str(i)].append( np.mean(torch.stack(list_mean_probs["1"]["step" + str(i)]).cpu().numpy()))
        plt.plot(list_mean_probs_history_val["1"]["step" + str(i)], '-', label="Class 1 Iter" + str(i))
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Mean Probability per Class per Epoch Validation.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(1)
    plt.plot(val_loss_in_history, label='Loss')

    plt.ylabel('Loss'), plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Validation Loss per Iteration.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(1)
    plt.plot(val_prec1_in_history, 'g', label='Precision class 1')
    plt.plot(val_prec0_in_history, 'r', label='Precision class 0')
    plt.ylabel('Precision'), plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Validation Precision per Iteration.pdf', bbox_inches='tight')
    plt.close()

    return val_losses, val_losses1, val_losses0, val_precision_1, val_precision_0,val_loss_in_history,val_prec1_in_history, val_prec0_in_history,val_prec_in_history,list_mean_probs_history_val, val_FPR




# # CODIGO PINTAS DIST Y DIST NORM (va en el bucle)

    # spatial_dist_g_l = []
    #         spatial_dist_g_l_norm = []
    #         pets_dists = []
    #         pets_dists_norm = []
    #         terrace_dists = []
    #         terrace_dists_norm = []
    #         lab_dists = []
    #         lab_dists_norm = []
    # basket_dists  = []
    #         garden1_dists = []
    #         garden1_dists_norm = []


# spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))), dim=1).cuda()
#                 spatial_dist_g_l.append(spatial_dist_g.cpu().numpy())
#                 spatial_dist_g_norm.append(spatial_dist_g.cpu().numpy() / max_dist[g])
#                 spatial_dist_x = torch.abs(torch.from_numpy(xws_1 - xws_2)).cuda()
#                 spatial_dist_x_norm = spatial_dist_x / max_dist[g]
#                 spatial_dist_y = torch.abs(torch.from_numpy(yws_1 - yws_2)).cuda()
#                 spatial_dist_y_norm = spatial_dist_y / max_dist[g]
# #
# edge_labels_g = torch.from_numpy(
#     np.asarray([1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
#                       data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
#                 for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()
# if max_dist[g] == 26.56:  # PETS
#     pets_dists.append(
#         [n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if edge_labels_g.cpu().numpy()[pos] == 1])
#     pets_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                             edge_labels_g.cpu().numpy()[pos] == 1])
# elif max_dist[g] == 50.83:  # Terrace
#     terrace_dists.append([n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                           edge_labels_g.cpu().numpy()[pos] == 1])
#     terrace_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                                edge_labels_g.cpu().numpy()[pos] == 1])
# elif max_dist[g] == 44.23:  # Laboratory
#     lab_dists.append([n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                       edge_labels_g.cpu().numpy()[pos] == 1])
#     lab_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                            edge_labels_g.cpu().numpy()[pos] == 1])
# elif max_dist[g] == 85.23:  # Garden1 CAMPUS
#     garden1_dists.append([n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                           edge_labels_g.cpu().numpy()[pos] == 1])
#     garden1_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                                edge_labels_g.cpu().numpy()[pos] == 1])



# # CODIGO PINTAR DISTSN desopues de forwards
#  dists = np.concatenate(spatial_dist_g_l)
#             dists_norm = np.concatenate(spatial_dist_g_l_norm)
#
#             pets_dists = np.concatenate(pets_dists)
#             pets_dists_norm = np.concatenate(pets_dists_norm)
#             terrace_dists = np.concatenate(terrace_dists)
#             terrace_dists_norm = np.concatenate(terrace_dists_norm)
#             lab_dists = np.concatenate(lab_dists)
#             lab_dists_norm = np.concatenate(lab_dists_norm)
#             garden1_dists = np.concatenate(garden1_dists)
#             garden1_dists_norm = np.concatenate(garden1_dists_norm)
#
#
#             pets_dists_mean = np.mean(pets_dists)
#             pets_dists_norm_mean = np.mean(pets_dists_norm)
#             terrace_dists_mean = np.mean(terrace_dists)
#             terrace_dists_norm_mean = np.mean(terrace_dists_norm)
#             lab_dists_norm_mean = np.mean(lab_dists_norm)
#             lab_dists_mean = np.mean(lab_dists)
#             garden1_dists_mean = np.mean(garden1_dists)
#             garden1_dists_norm_mean = np.mean(garden1_dists_norm)
#
#
#             plt.figure()
#             plt.subplot(2, 1, 1)
#             plt.scatter(np.arange(len(dists)), dists, c=data_batch.edge_labels.cpu().numpy())
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * terrace_dists_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * pets_dists_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * lab_dists_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * garden1_dists_mean)
#
#
#             plt.title('Distances. Mean(1) terrace = ' + str(int(terrace_dists_mean)) + ' Mean(1) pets = ' + str(
#                 int(pets_dists_mean)) + 'Mean(1) Lab = ' + str(int(lab_dists_mean)) + 'Mean(1) Garden1 = ' + str(int(garden1_dists_mean)) )
#             plt.show(block=False)
#             plt.subplot(2, 1, 2)
#             plt.scatter(np.arange(len(dists_norm)), dists_norm, c=data_batch.edge_labels.cpu().numpy())
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * terrace_dists_norm_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * pets_dists_norm_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * lab_dists_norm_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * garden1_dists_norm_mean)
#
#
#             plt.title(
#                 'Distances in meters. Mean(1) terrace = ' + str((terrace_dists_norm_mean)) + ' Mean(1) pets = ' + str(
#                     pets_dists_norm_mean) + 'Mean(1) Lab = ' + str(lab_dists_norm_mean) + 'Mean(1) Garden1 = ' + str(garden1_dists_norm_mean) )
#
#             plt.show(block=False)


# dists = np.concatenate(spatial_dist_g_l)
# dists_norm = np.concatenate(spatial_dist_g_l_norm)
# basket_dists = np.concatenate(basket_dists)
# basket_dists_norm = np.concatenate(basket_dists_norm)
# basket_dists_mean = np.mean(basket_dists)
# basket_dists_norm_mean = np.mean(basket_dists_norm)
#
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.scatter(np.arange(len(dists)), dists, c=data_batch.edge_labels.cpu().numpy())
# plt.plot(np.arange(len(dists)), np.ones(len(dists)) * basket_dists_mean)
#
#
# plt.title('Distances. Mean(1) Basketball = ' + str(int(basket_dists_mean))  )
# plt.subplot(2, 1, 2)
# plt.scatter(np.arange(len(dists_norm)), dists_norm, c=data_batch.edge_labels.cpu().numpy())
# plt.plot(np.arange(len(dists)), np.ones(len(dists)) * basket_dists_norm_mean)
#
# plt.title(  'Distances in meters. Mean(1) Basketball = ' + str((basket_dists_norm_mean)) )
#
# plt.show(block=False)