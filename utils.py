
import os
import os.path as osp
import warnings

import sys
import cv2
import math
import time
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.spatial.distance import cdist
import torch
import shutil
import yaml
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt
from torch_scatter import scatter_add
from torch.nn import functional as F
import torch.nn as nn
from typing import Optional


def intersect(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]
def compute_SCC_and_Clusters(G,n_nodes):
    sets = [c for c in sorted(nx.strongly_connected_components(G), key=len, reverse=False)]

    # Add independent nodes to the list of CCs
    for i in range(n_nodes):
        flag = 0
        for s in sets:
            s = list(s)
            if i in s:
                flag = 1
                break
        if flag == 0:
            sets.append({i})

    ID_pred = torch.zeros( n_nodes, dtype=int)
    n_components_pred = sets.__len__()
    cluster = 0
    for s in sets:
        for i in list(s):
            ID_pred[i] = cluster
        cluster = cluster + 1
    #
    return ID_pred, n_components_pred

def splitting(ID_pred, predictions, preds_prob, edge_list,data_batch,predicted_act_edges,num_cameras):
    # Count how many items in each class, then look which cluster label is the one with more than 4 elements (# cameras)

    label_ID_to_disjoint = torch.where(torch.bincount(ID_pred) > num_cameras)[0]

    # label_ID_to_disjoint = np.where(np.bincount(ID_pred) > num_cameras)[0]
    if len(label_ID_to_disjoint) >= 1:
        # global_idx_new_predicted_active_edges = [pos for pos, p in enumerate(predictions) if p == 1]

        # COMPROBAR SI HAY DOS
        l = label_ID_to_disjoint[0]
        flag_need_disjoint = True
        while flag_need_disjoint:
            # global_idx_new_predicted_active_edges = [pos for pos, p in enumerate(predictions) if p == 1]
            global_idx_new_predicted_active_edges = (predictions == 1).nonzero(as_tuple=True)[0]
            nodes_to_disjoint = torch.where(ID_pred == l)[0]

            idx_active_edges_to_disjoint = [pos for pos, n in enumerate(predicted_act_edges) if    np.any(np.in1d(nodes_to_disjoint, n))]

            # OPTION: REMOVING BRIDGES
            # bridges = list(nx.bridges(nx.to_undirected(G)))
            # bridges = bridges + [n[::-1] for n in bridges]
            # # candidates_edges_disjoint = [predicted_act_edges[p] for p in idx_active_edges_to_disjoint]
            #
            # if len(bridges) > 0:
            #     a = 1
            #     idx_bridges = [predicted_act_edges.index(n) for pos, n in enumerate(bridges)]
            #     global_idx_bridges = np.asarray(global_idx_new_predicted_active_edges)[ np.asarray(idx_bridges)]
            #     min_prob = np.min(preds_prob[global_idx_bridges].cpu().numpy())
            #     global_idx_min_prob = np.where(preds_prob.cpu().numpy() == min_prob)[0]
            #     predictions[global_idx_min_prob] = 0
            #
            # if len(bridges) == 0:
            #     global_idx_edges_disjoint = np.asarray(global_idx_new_predicted_active_edges)[
            #         np.asarray(idx_active_edges_to_disjoint)]
            #     min_prob = np.min(preds_prob[global_idx_edges_disjoint].cpu().numpy())
            #     global_idx_min_prob = np.where(preds_prob.cpu().numpy() == min_prob)[0]
            #     predictions[global_idx_min_prob] = 0

            # OPTION: REMOVING MINIMUM PROB EDGES
            # global_idx_edges_disjoint = np.asarray(global_idx_new_predicted_active_edges)[np.asarray(idx_active_edges_to_disjoint)]
            global_idx_edges_disjoint = global_idx_new_predicted_active_edges[ torch.tensor(idx_active_edges_to_disjoint)]
            min_prob = torch.min(preds_prob[global_idx_edges_disjoint])
            global_idx_min_prob = torch.where(preds_prob == min_prob)[0]
            predictions[global_idx_min_prob] = 0


            # Check if still need to disjoint
            # t = time.tic()
            # predicted_act_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(predictions) if p == 1]
            # print('time: ' str(time.tic()-t))
            predicted_act_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos in torch.where(predictions == 1)[0]]



            G = nx.DiGraph(predicted_act_edges)
            ID_pred, n_clusters_pred = compute_SCC_and_Clusters(G,   data_batch.num_nodes)

            if np.bincount(ID_pred)[l] > num_cameras:
                flag_need_disjoint = True
                # predictions, predicted_act_edges = remove_edges_single_direction(predicted_act_edges, predictions, edge_list)
                # G = nx.DiGraph(predicted_act_edges) #COMPROBAR
            else:
                flag_need_disjoint = False
                # predictions,predicted_act_edges =  remove_edges_single_direction(predicted_act_edges, predictions, edge_list)
                # G = nx.DiGraph(predicted_act_edges)

                splitting(ID_pred, predictions, preds_prob, edge_list, data_batch, predicted_act_edges,num_cameras)

    return predictions

def remove_edges_single_direction(active_edges, predictions, edge_list):
    idx_active_edges_to_remove = [pos for pos, n in enumerate(active_edges) if (n[::-1] not in active_edges)]
    if idx_active_edges_to_remove != []:
        predicted_active_edges_global_pos = [pos for pos, p in enumerate(predictions) if p == 1]

        global_idx_edges_to_remove = np.asarray(predicted_active_edges_global_pos)[np.asarray(idx_active_edges_to_remove)]
        new_predictions = predictions.clone()
        new_predictions[global_idx_edges_to_remove] = 0

        new_predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                      enumerate(new_predictions) if p == 1]

    else:
        new_predictions = predictions.clone()
        new_predicted_active_edges = active_edges


    return new_predictions, new_predicted_active_edges

def pruning(graph_obj, edges_out, probs,predicted_active_edges,num_cameras):
    """
    Determines the proportion of Flow Conservation inequalities that are satisfied.
    For each node, the sum of incoming (resp. outgoing) edge values must be less or equal than 1.

    Args:
        graph_obj: 'Graph' object
        edges_out: BINARIZED output values for edges (1 if active, 0 if not active)
        undirected_edges: determines whether each edge in graph_obj.edge_index appears in both directions (i.e. (i, j)
        and (j, i) are both present (undirected_edges =True), or only (i, j), with  i<j (undirected_edges=False)
        return_flow_vals: determines whether the sum of incoming /outglong flow for each node must be returned

    Returns:
        constr_sat_rate: float between 0 and 1 indicating the proprtion of inequalities that are satisfied

    """
    # Get tensors indicataing which nodes have incoming and outgoing flows (e.g. nodes in first frame have no in. flow)
    undirected_edges = False
    edge_ixs = graph_obj.edge_index
    if undirected_edges:
        sorted, _ = edge_ixs.t().sort(dim = 1)
        sorted = sorted.t()
        div_factor = 2. # Each edge is predicted twice, hence, we divide by 2
    else:
        sorted = edge_ixs # Edges (i.e. node pairs) are already sorted
        div_factor = 1.  # Each edge is predicted once, hence, hence we divide by 1.

    flag_rounding_needed = False
    # Compute incoming and outgoing flows for each node
    flow_out = scatter_add(edges_out, sorted[0],dim_size=graph_obj.num_nodes) / div_factor
    flow_in = scatter_add(edges_out, sorted[1], dim_size=graph_obj.num_nodes) / div_factor


    # nodes_flow_out = np.where(flow_out.cpu().numpy() > 3)
    nodes_flow_out = torch.where(flow_out > (num_cameras-1))
    #
    # nodes_flow_in = np.where(flow_in.cpu().numpy() > 3)
    nodes_flow_in = torch.where(flow_in > (num_cameras-1))


    if (len(nodes_flow_out[0]) != 0 or len(nodes_flow_in[0]) != 0):
        flag_rounding_needed = True
        new_predictions = edges_out.clone()
    else:
        new_predictions = []

    while flag_rounding_needed:
        edges_to_remove = []

        # METHOD BRIDGES
        #
        # G = nx.DiGraph(predicted_active_edges)
        # bridges = list(nx.bridges(nx.to_undirected(G)))
        # bridges = bridges + [n[::-1] for n in bridges]

        # if len(bridges) == 0:
        #     for n in nodes_flow_out[0]:
        #         # ## CPU
        #         # pos =np.intersect1d(np.where(edge_ixs.cpu().numpy()[0] == n), np.where(new_predictions.cpu().numpy()==1)[0])
        #         # remove_edge = pos[np.argmin(probs[pos].cpu().numpy())]
        #         # edges_to_remove.append(remove_edge)
        #
        #         pos = intersect(torch.where(edge_ixs[0] == n)[0], torch.where(new_predictions == 1)[0])
        #         remove_edge = pos[torch.argmin(probs[pos])]
        #         edges_to_remove.append(remove_edge)
        #
        #
        #     for n in nodes_flow_in[0]:
        #         # ## CPU
        #
        #         # pos = np.intersect1d(np.where(edge_ixs.cpu().numpy()[1] == n), np.where(new_predictions.cpu().numpy()==1)[0])
        #         # remove_edge = pos[np.argmin(probs[pos].cpu().numpy())]
        #         # edges_to_remove.append(remove_edge)
        #
        #         pos = intersect(torch.where(edge_ixs[1] == n)[0], torch.where(new_predictions == 1)[0])
        #         remove_edge = pos[torch.argmin(probs[pos])]
        #         edges_to_remove.append(remove_edge)
        #
        # else:  # if there are bridges
        #     a=1
        #     for n in nodes_flow_out[0]:
        #         # ## CPU
        #         # pos =np.intersect1d(np.where(edge_ixs.cpu().numpy()[0] == n), np.where(new_predictions.cpu().numpy()==1)[0])
        #         # edge_tuple = list(map(tuple, np.transpose(edge_ixs.cpu().numpy())))
        #
        #         pos = intersect(torch.where(edge_ixs[0] == n)[0], torch.where(new_predictions == 1)[0])
        #         # edge_tuple = list(map(tuple, torch.transpose(edge_ixs)))
        #         edge_tuple = list(map(tuple, np.transpose(edge_ixs.cpu().numpy())))
        #
        #
        #         pos_bridges = [p for p, tp in enumerate(edge_tuple) if tp in bridges]
        #
        #         for p, na in enumerate(pos):
        #             if na in pos_bridges:
        #                 edges_to_remove.append(na)
        #
        #     for n in nodes_flow_in[0]:
        #         # ## CPU
        #         # pos = np.intersect1d(np.where(edge_ixs.cpu().numpy()[1] == n), np.where(new_predictions.cpu().numpy()==1)[0])
        #         # edge_tuple = list(map(tuple, np.transpose(edge_ixs.cpu().numpy())))
        #
        #         pos = intersect(torch.where(edge_ixs[1] == n)[0], torch.where(new_predictions == 1)[0])
        #         edge_tuple = list(map(tuple, np.transpose(edge_ixs.cpu().numpy())))
        #
        #         pos_bridges = [p for p, tp in enumerate(edge_tuple) if tp in bridges]
        #
        #         for p, na in enumerate(pos):
        #             if na in pos_bridges:
        #                 edges_to_remove.append(na)
        #
        #     if edges_to_remove == []:  # do it regularly
        #         for n in nodes_flow_out[0]:
        #             # ## CPU
        #             # pos = np.intersect1d(np.where(edge_ixs.cpu().numpy()[0] == n), np.where(new_predictions.cpu().numpy() == 1)[0])
        #             # remove_edge = pos[np.argmin(probs[pos].cpu().numpy())]
        #
        #             pos = intersect(torch.where(edge_ixs[0] == n)[0], torch.where(new_predictions == 1)[0])
        #             remove_edge = pos[torch.argmin(probs[pos])]
        #
        #             edges_to_remove.append(remove_edge)
        #
        #         for n in nodes_flow_in[0]:
        #             # ## CPU
        #             # pos = np.intersect1d(np.where(edge_ixs.cpu().numpy()[1] == n),   np.where(new_predictions.cpu().numpy() == 1)[0])
        #             # remove_edge = pos[np.argmin(probs[pos].cpu().numpy())]
        #
        #             pos = intersect(torch.where(edge_ixs[1] == n)[0], torch.where(new_predictions == 1)[0])
        #             remove_edge = pos[torch.argmin(probs[pos])]
        #             edges_to_remove.append(remove_edge)


        # METHOD NO BRIGDES
        # print('NO BRIDGES')
        t = time.time()
        for n in nodes_flow_out[0]:
            # ## CPU
            # pos = np.intersect1d(np.where(edge_ixs.cpu().numpy()[0] == n), np.where(new_predictions.cpu().numpy() == 1)[0])
            # remove_edge = pos[np.argmin(probs[pos].cpu().numpy())]

            pos = intersect(torch.where(edge_ixs[0] == n)[0], torch.where(new_predictions == 1)[0])
            remove_edge = pos[torch.argmin(probs[pos])]

            edges_to_remove.append(remove_edge)

        for n in nodes_flow_in[0]:
            # ## CPU
            # pos = np.intersect1d(np.where(edge_ixs.cpu().numpy()[1] == n),   np.where(new_predictions.cpu().numpy() == 1)[0])
            # remove_edge = pos[np.argmin(probs[pos].cpu().numpy())]

            pos = intersect(torch.where(edge_ixs[1] == n)[0], torch.where(new_predictions == 1)[0])
            remove_edge = pos[torch.argmin(probs[pos])]
            edges_to_remove.append(remove_edge)

        edges_to_remove = torch.stack(edges_to_remove)

        # #FIN
        #
        #
        if edges_to_remove.shape[0] >= 1:
            new_predictions[edges_to_remove] = 0
        else:
            new_predictions = []

        #COMPUTE FLOW AGAIN TO CHECK
        flow_out = scatter_add(new_predictions, sorted[0], dim_size=graph_obj.num_nodes) / div_factor
        flow_in = scatter_add(new_predictions, sorted[1], dim_size=graph_obj.num_nodes) / div_factor

        nodes_flow_out = np.where(flow_out.cpu().numpy() > (num_cameras-1))
        nodes_flow_in = np.where(flow_in.cpu().numpy() > (num_cameras-1))

        if (len(nodes_flow_out[0]) != 0 or len(nodes_flow_in[0]) != 0):
            flag_rounding_needed = True
        else:
            flag_rounding_needed = False



    # Determine how many inequalitites are violated
    # violated_flow_out = (flow_out > 3).sum()
    # violated_flow_in = (flow_in > 3).sum()

    # Compute the final constraint satisfaction rate
    # violated_inequalities = (violated_flow_in + violated_flow_out).float()
    # flow_out_constr, flow_in_constr= sorted[0].unique(), sorted[1].unique()
    # num_constraints = len(flow_out_constr) + len(flow_in_constr)
    # constr_sat_rate = 1 - violated_inequalities / num_constraints
    # if constr_sat_rate.item() < 1:
    #     a=1
    #     # np.asarray([(edge_ixs[0][pos], edge_ixs[1][pos]) for pos, j in enumerate(edges_out) if j.item() == 1])
    # if not return_flow_vals:
    #     return constr_sat_rate.item(), new_predictions
    #
    # else:
    #     return constr_sat_rate.item(), flow_in, flow_out

    return new_predictions

def visualize(h, color, edge_labels = None,edge_index =None ,node_label = None,epoch=None, loss=None, mode = None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    pos = nx.spring_layout(h, seed=42)
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=20, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    elif edge_labels is None:

        nx.draw_networkx(h, pos=pos, with_labels=False,
                         node_color=color, cmap="Set3")

    elif edge_index is not None:
        list_active_edges = [(edge_index[0][i], edge_index[1][i]) for i in range(len(edge_labels)) if
                             edge_labels[i] == 1]
        list_nonactive_edges = [(edge_index[0][i], edge_index[1][i]) for i in range(len(edge_labels)) if
                                edge_labels[i] == 0]
        nx.draw_networkx_nodes(h, pos=pos,node_color=color,cmap="Set2")
        if node_label is not None:
            list_node_label = {n: node_label[n] for n in range(len(node_label))}
            nx.draw_networkx_labels(h, pos, labels=list_node_label, font_size=16)

        nx.draw_networkx_edges(h, pos=pos, edgelist=list_active_edges, edge_color='darkred')
        nx.draw_networkx_edges(h, pos=pos, edgelist=list_nonactive_edges, edge_color='lightgray',alpha=0.5)
        # nx.draw_networkx_edges(h, pos=pos, edge_list=edges_ixs_nonactives, edge_color='b')
    # elif node_label is not None:
    #     list_node_label = {n: node_label[n] for n in range(len(node_label))}
    #     nx.draw_networkx_labels(h, pos, labels=list_node_label, font_size=16)

     # plt.show(block=False)
    plt.savefig('graph_'+ mode +'.png')
    a=1

def apply_homography_image_to_world(xi, yi, H_image_to_world):
    # Spatial vector xi, yi, 1
    S = np.array([xi, yi, 1]).reshape(3, 1)
    # Get 3 x 3 matrix and compute inverse
    # H_world_to_image = np.array(H_world_to_image).reshape(3, 3)
    # H_image_to_world = np.linalg.inv(H_world_to_image)

    # H_image_to_world = np.array(H_image_to_world).reshape(3, 3)

    # Dot product
    prj = np.dot(H_image_to_world, S)
    # Get world coordinates
    xw = (prj[0] / prj[2]).item() # latitude
    yw = (prj[1] / prj[2]).item() # longitude
    return xw, yw

def apply_homography_world_to_image(xi, yi, H_world_to_image):
    # Spatial vector xi, yi, 1
    S = np.array([xi, yi, 1]).reshape(3, 1)
    # Get 3 x 3 matrix and compute inverse
    H_world_to_image = np.array(H_world_to_image).reshape(3, 3)

    # Dot product
    prj = np.dot(H_world_to_image, S)
    # Get world coordinates
    xw = (prj[0] / prj[2]).item() # latitude
    yw = (prj[1] / prj[2]).item() # longitude
    return xw, yw

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(state, is_best, path):
    torch.save(state, path + '/models/' +  'latest.pth.tar')

    if is_best:
        print('Best model updated.')
        shutil.copyfile(path + '/models/' +  'latest.pth.tar',
                        path + '/models/' + 'best.pth.tar')
        # shutil.copyfile('config/config_training.yaml', path + '/files/config.yaml')

        dict_file = {'VALIDATION': {'ACCURACY_AVG': str(round(state['prec'], 2)) + ' %',
                               'ACCURACY_1': str(round(state['prec1'][-1], 2)) + ' %',
                               'ACCURACY_0': str(round(state['prec0'][-1], 2)) + ' %'},
                     'EPOCH': state['epoch'],
                     'MODEL PARAMETERS': str(state['model_parameters']) + ' Millions',
                     'DATASET': state['CONFIG']['DATASET']['data_train'],
                     'VAL_LOSS': state['best_loss']}

        with open(os.path.join(path, 'Summary Report.yaml'), 'w') as file:
            yaml.safe_dump(dict_file, file)

    #
    # dict_mean_file = {'Last ten epochs avg training accuracy': str(state['10epoch_train_prec']),
    #                   'Last ten epochs avg testing accuracy': str(state['10epoch_test_prec'])
    #                   }
    # with open(os.path.join(path, 'Average precisions.yaml'), 'w') as file:
    #     yaml.safe_dump(dict_mean_file, file)

def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict


    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(fpath, map_location=map_location)

    return checkpoint

def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::

    """
    checkpoint = load_checkpoint(weight_path)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded '
                  'due to unmatched keys or layer size: {}'.format(discarded_layers))


    return model

#  FROM TOP DB
def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::

    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input1.dim())
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input2.dim())
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat

def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    distmat = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat

def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def save_graphics(results_path,training_precision_1_avg,training_precision_0_avg,val_precision_1_avg,val_precision_0_avg,training_loss_avg, training_loss_avg_1,
                        training_loss_avg_0,val_loss_avg,val_loss_avg_1,val_loss_avg_0,list_lr, training_FPR_avg, val_FPR_avg):
    plt.figure()
    plt.plot(training_precision_1_avg, label='Training Prec class 1')
    plt.plot(training_precision_0_avg, label='Training Prec class 0')
    plt.plot(val_precision_1_avg, label='Validation Prec class 1')
    plt.plot(val_precision_0_avg, label='Validation Prec class 0')
    plt.plot((np.asarray(training_precision_1_avg) + np.asarray(training_precision_0_avg)) / 2, '--',   label='Training MCA')
    plt.plot((np.asarray(val_precision_1_avg) + np.asarray(val_precision_0_avg)) / 2, '--', label='Validation MCA')
    plt.ylabel('Precision'), plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(results_path + '/images/Precision Per Epoch.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(training_loss_avg, c='red',label='Training loss')
    plt.plot(training_loss_avg_1, '--',c='firebrick' ,label='Training loss 1')
    plt.plot(training_loss_avg_0, '--',c = 'indianred', label='Training loss 0')
    plt.plot(val_loss_avg, c='green', label='Validation loss')
    plt.plot(val_loss_avg_1,'--', c='darkgreen',label='Validation loss 1')
    plt.plot(val_loss_avg_0, '--',c='limegreen',label='Validation loss 0')

    plt.ylabel('Loss'), plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(results_path + '/images/ All Losses per Epoch.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(training_loss_avg, c='red', label='Training loss')
    plt.plot(val_loss_avg, c='green', label='Validation loss')
    plt.ylabel('Loss'), plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(results_path + '/images/Loss per Epoch.pdf', bbox_inches='tight')
    plt.close()


    plt.figure()
    plt.plot(list_lr, 'r', label='Training')
    plt.ylabel('Learning Rate'), plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(results_path + '/images/LR per Epoch.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(training_FPR_avg, c='red', label='Training FPR')
    plt.plot(val_FPR_avg, c='green', label='Validation FPR')
    plt.ylabel('FPR'), plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(results_path + '/images/FPR per Epoch.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    fig, axs = plt.subplots(5, sharex=True)

    fig.set_size_inches(6, 10)

    # fig.suptitle(CONFIG['MODEL']['ARCH'] + ' ' + CONFIG['DATASET']['NAME'] + ' ' + CONFIG['DISTILLATION']['D_LOSS'])

    axs[0].plot(training_loss_avg, c='red', label='Training loss')
    axs[0].plot(val_loss_avg, c='green', label='Validation loss')
    axs[0].set(ylabel='Loss')
    axs[0].legend()

    axs[1].plot(training_loss_avg, c='red',label='Training loss')
    axs[1].plot(training_loss_avg_1, '--',c='firebrick' ,label='Training loss 1')
    axs[1].plot(training_loss_avg_0, '--',c = 'indianred', label='Training loss 0')
    axs[1].plot(val_loss_avg, c='green', label='Validation loss')
    axs[1].plot(val_loss_avg_1,'--', c='darkgreen',label='Validation loss 1')
    axs[1].plot(val_loss_avg_0, '--',c='limegreen',label='Validation loss 0')
    axs[1].legend()
    axs[1].set(ylabel='Loss')

    axs[2].plot(training_precision_1_avg, label='Training Prec class 1')
    axs[2].plot(training_precision_0_avg, label='Training Prec class 0')
    axs[2].plot(val_precision_1_avg, label='Validation Prec class 1')
    axs[2].plot(val_precision_0_avg, label='Validation Prec class 0')
    axs[2].plot((np.asarray(training_precision_1_avg) + np.asarray(training_precision_0_avg)) / 2, '--',   label='Training MCA')
    axs[2].plot((np.asarray(val_precision_1_avg) + np.asarray(val_precision_0_avg)) / 2, '--', label='Validation MCA')
    axs[2].set(ylabel='Precision')
    axs[2].legend()

    axs[3].plot(list_lr, 'r', label='Training')
    axs[3].set(ylabel='Learning Rate')
    axs[3].legend()


    axs[4].plot(training_FPR_avg, c='red', label='Training FPR')
    axs[4].plot(val_FPR_avg, c='green', label='Validation FPR')
    axs[4].set(xlabel='Epoch', ylabel='FPR')
    axs[4].legend()

    plt.savefig(results_path + '/images/Summary figure.pdf', bbox_inches='tight', dpi = 300)
    plt.close()


class FocalLoss_binary(nn.Module):
    """
    Class definition for the Focal Loss. Extracted from the paper Focal Loss for Dense Object detection by FAIR.
    """

    def __init__(self, focusing_param=5, balance_param=0.9, reduction = 'mean'):
        super(FocalLoss_binary, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param
        self.cross_entropy = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, output, target):
        """
        Computes the focal loss for a classification problem (scene classification)
        :param output: Output obtained by the network
        :param target: Ground-truth labels
        :return: Focal loss
        """
        # Compute the regular cross entropy between the output and the target
        logpt = - self.cross_entropy(output,target)
        # Compute pt
        pt = torch.exp(logpt)

        # Compute focal loss
        focal_loss = -((1 - pt) ** self.focusing_param) * logpt
        # Apply weighting factor to obtain balanced focal loss
        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss



##### FROM https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
 #DA CUDA OYT OF MEMORY CUANDO RED = NONE
# class FocalLoss(nn.CrossEntropyLoss):
#     ''' Focal loss for classification tasks on imbalanced datasets '''
#
#     def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
#         super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
#         self.reduction = reduction
#         self.gamma = gamma
#
#     def forward(self, input_, target):
#         cross_entropy = super().forward(input_, target)
#         # Temporarily mask out ignore index to '0' for valid gather-indices input.
#         # This won't contribute final loss as the cross_entropy contribution
#         # for these would be zero.
#         target = target * (target != self.ignore_index).long()
#         input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
#         loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
#         return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum'  else loss



##FROM https://github.com/namdvt/Focal-loss-pytorch-implementation/blob/master/function.py


# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         bce_loss = F.binary_cross_entropy(inputs.squeeze(),  targets.float())
#         loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
#         return loss
#


#### FROM https://github.com/kornia/kornia/blob/8520fa8366faa904619f0ecb84b1350236eccd54/kornia/utils/one_hot.py#L6
# def one_hot(
#     labels: torch.Tensor,
#     num_classes: int,
#     device: Optional[torch.device] = None,
#     dtype: Optional[torch.dtype] = None,
#     eps: float = 1e-6,
# ) -> torch.Tensor:
#     r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.
#     Args:
#         labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
#           Each value is an integer representing correct classification.
#         num_classes: number of classes in labels.
#         device: the desired device of returned tensor.
#         dtype: the desired data type of returned tensor.
#     Returns:
#         the labels in one hot tensor of shape :math:`(N, C, *)`,
#     Examples:
#         # >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
#         # >>> one_hot(labels, num_classes=3)
#         # tensor([[[[1.0000e+00, 1.0000e-06],
#                   [1.0000e-06, 1.0000e+00]],
#         <BLANKLINE>
#                  [[1.0000e-06, 1.0000e+00],
#                   [1.0000e-06, 1.0000e-06]],
#         <BLANKLINE>
#                  [[1.0000e-06, 1.0000e-06],
#                   [1.0000e+00, 1.0000e-06]]]])
#     """
#     if not isinstance(labels, torch.Tensor):
#         raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")
#
#     if not labels.dtype == torch.int64:
#         raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")
#
#     if num_classes < 1:
#         raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))
#
#     shape = labels.shape
#     one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)
#
#     return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
#
#
# def focal_loss(
#     input: torch.Tensor,
#     target: torch.Tensor,
#     alpha: float,
#     gamma: float = 1.5,
#     reduction: str = 'none',
#     eps: Optional[float] = None,
# ) -> torch.Tensor:
#     r"""Criterion that computes Focal loss.
#     According to :cite:`lin2018focal`, the Focal loss is computed as follows:
#     .. math::
#         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
#     Where:
#        - :math:`p_t` is the model's estimated probability for each class.
#     Args:
#         input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
#         target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
#         alpha: Weighting factor :math:`\alpha \in [0, 1]`.
#         gamma: Focusing parameter :math:`\gamma >= 0`.
#         reduction: Specifies the reduction to apply to the
#           output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
#           will be applied, ``'mean'``: the sum of the output will be divided by
#           the number of elements in the output, ``'sum'``: the output will be
#           summed.
#         eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.
#     Return:
#         the computed loss.
#     Example:
#         # >>> N = 5  # num_classes
#         # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
#         # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
#         # >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
#         # >>> output.backward()
#     """
#     if eps is not None and not torch.jit.is_scripting():
#         warnings.warn(
#             "`focal_loss` has been reworked for improved numerical stability "
#             "and the `eps` argument is no longer necessary",
#             DeprecationWarning,
#             stacklevel=2,
#         )
#
#     if not isinstance(input, torch.Tensor):
#         raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
#
#     if not len(input.shape) >= 2:
#         raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")
#
#     if input.size(0) != target.size(0):
#         raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')
#
#     n = input.size(0)
#     out_size = (n,) + input.size()[2:]
#     if target.size()[1:] != input.size()[2:]:
#         raise ValueError(f'Expected target size {out_size}, got {target.size()}')
#
#     if not input.device == target.device:
#         raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
#
#     # compute softmax over the classes axis
#     input_soft: torch.Tensor = F.softmax(input, dim=1)
#     log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)
#
#     # create the labels one hot tensor
#     target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)
#
#     # compute the actual focal loss
#     weight = torch.pow(-input_soft + 1.0, gamma)
#
#     focal = -alpha * weight * log_input_soft
#     loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))
#
#     if reduction == 'none':
#         loss = loss_tmp
#     elif reduction == 'mean':
#         loss = torch.mean(loss_tmp)
#     elif reduction == 'sum':
#         loss = torch.sum(loss_tmp)
#     else:
#         raise NotImplementedError(f"Invalid reduction mode: {reduction}")
#     return loss
#
# class FocalLoss(nn.Module):
#     r"""Criterion that computes Focal loss.
#     According to :cite:`lin2018focal`, the Focal loss is computed as follows:
#     .. math::
#         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
#     Where:
#        - :math:`p_t` is the model's estimated probability for each class.
#     Args:
#         alpha: Weighting factor :math:`\alpha \in [0, 1]`.
#         gamma: Focusing parameter :math:`\gamma >= 0`.
#         reduction: Specifies the reduction to apply to the
#           output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
#           will be applied, ``'mean'``: the sum of the output will be divided by
#           the number of elements in the output, ``'sum'``: the output will be
#           summed.
#         eps: Deprecated: scalar to enforce numerical stability. This is no longer
#           used.
#     Shape:
#         - Input: :math:`(N, C, *)` where C = number of classes.
#         - Target: :math:`(N, *)` where each value is
#           :math:`0 ≤ targets[i] ≤ C−1`.
#     Example:
#         # >>> N = 5  # num_classes
#         # >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
#         # >>> criterion = FocalLoss(**kwargs)
#         # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
#         # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
#         # >>> output = criterion(input, target)
#         # >>> output.backward()
#     """
#
#     def __init__(self, alpha: float, gamma: float =0, reduction: str = 'none', eps: Optional[float] = None) -> None:
#         super().__init__()
#         self.alpha: float = alpha
#         self.gamma: float = gamma
#         self.reduction: str = reduction
#         self.eps: Optional[float] = eps
#
#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)
