
import os
import time
import shutil
import yaml
import datetime


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import networkx as nx
from sklearn.metrics.pairwise import paired_distances
from sklearn import metrics
import utils
from torch_geometric.utils import to_networkx
import dtw
list_cam_colors = list(['royalblue', 'darkorange','green','firebrick'])



def compute_P_R_F(preds, labels):
    index_label_1 = torch.where(labels == 1)[0]
    index_label_0 = torch.where(labels == 0)[0]
    precision_class1 = list()
    precision_class0 = list()
    ## Add computation of precision of class 1 and class 0
    sum_successes_1 = torch.sum(preds[index_label_1] == labels[index_label_1])
    if sum_successes_1 == 0:
        precision_class1.append(torch.tensor(0.0).cuda())
    else:
        precision_class1.append((sum_successes_1 / len(labels[index_label_1])) * 100.0)

    # Precision class 0
    sum_successes_0 = torch.sum(preds[index_label_0] == labels[index_label_0])
    if sum_successes_0 == 0:
        precision_class0.append(torch.tensor(0.0).cuda())
    else:
        precision_class0.append((sum_successes_0 / len(labels[index_label_0])) * 100.0)

    ##

    # TP
    TP = torch.sum(preds[index_label_1] == 1)
    # FP
    FP = torch.sum(preds[index_label_0] == 1)
    # TN
    TN = torch.sum(preds[index_label_0] == 0)
    # FN
    FN = torch.sum(preds[index_label_1] == 0)
    # P, R , Fscore
    if (TP + FP) != 0:
        P = TP / (TP + FP)
    else:
        P =torch.tensor(0.0).cuda()

    if (TP + FN) != 0:
        R = TP / (TP + FN)
    else:
        R = torch.tensor(0.0).cuda()

    if (P + R) != 0:
        F = 2 * (P * R) / (P + R)
    else:
        F =torch.tensor(0.0).cuda()

    return TP, FP, TN, FN, P,R, F, precision_class0, precision_class1

def post_processing(num_cameras, ID_pred,predicted_active_edges, predictions, edge_list, CONFIG , data, preds_prob):
    print('Start preprocessing')
    num_CC = len(np.unique(ID_pred))
    print('# CC = ' + str(num_CC))

    if type(CONFIG['CUTTING']) is str:
        if CONFIG['CUTTING'] == 'True':
            CONFIG['CUTTING'] = True
        else:
            CONFIG['CUTTING'] = False

    if type(CONFIG['PRUNING']) is str:
        if CONFIG['PRUNING'] == 'True':
            CONFIG['PRUNING'] = True
        else:
            CONFIG['PRUNING'] = False

    if type(CONFIG['SPLITTING']) is str:
        if CONFIG['SPLITTING'] == 'True':
            CONFIG['SPLITTING'] = True
        else:
            CONFIG['SPLITTING'] = False

    if CONFIG['CUTTING']:
        print('First cutting')
        t = time.time()
        predictions, predicted_active_edges = utils.remove_edges_single_direction(predicted_active_edges,
                                                                                  predictions, edge_list)
        G = nx.DiGraph(predicted_active_edges)
        ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G, data.num_nodes)
        # print('# Clusters CUT:  ' + str(n_clusters_pred))
        # print(ID_pred)

        elapsed = time.time() - t
        # print('Elapsed first cutting : ' + str(elapsed))
        num_CC = len(torch.unique(ID_pred))
        print('# CC = ' + str(num_CC))


    # # COMPUTE GREEDY ROUNDING
    if CONFIG['PRUNING']:
        print('Pruning')
        t = time.time()
        predictions_r = utils.pruning(data, predictions.view(-1), preds_prob[:, 1],
                                               predicted_active_edges, num_cameras)
        if predictions_r != []:
            predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                      enumerate(predictions_r) if p == 1]
            predictions = predictions_r

        else:
            predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                      enumerate(predictions) if p == 1]

        G = nx.DiGraph(predicted_active_edges)
        ID_pred, rounding_n_clusters_pred = utils.compute_SCC_and_Clusters(G, data.num_nodes)
        # print('# Clusters with rounding:  ' + str(rounding_n_clusters_pred))
        # print(ID_pred)

        elapsed = time.time() - t
        # print('Elapsed prunning : ' + str(elapsed))
        num_CC = len(torch.unique(ID_pred))
        print('# CC = ' + str(num_CC))

    if CONFIG['CUTTING']:
        print('Second cutting')
        t = time.time()

        predictions, predicted_active_edges = utils.remove_edges_single_direction(predicted_active_edges,
                                                                                  predictions, edge_list)
        G = nx.DiGraph(predicted_active_edges)
        ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G, data.num_nodes)
        # print('# Clusters CUT:  ' + str(n_clusters_pred))
        # print(ID_pred)

        elapsed = time.time() - t
        # print('Elapsed second cutting : ' + str(elapsed))
        num_CC = len(torch.unique(ID_pred))
        print('# CC = ' + str(num_CC))


    if CONFIG['SPLITTING']:
        t = time.time()
        print('Splitting')
        predictions = utils.splitting(ID_pred, predictions, preds_prob[:, 1], edge_list,
                                                  data, predicted_active_edges,  num_cameras)
        predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                  enumerate(predictions) if p == 1]
        G = nx.DiGraph(predicted_active_edges)
        ID_pred, disjoint_n_clusters_pred = utils.compute_SCC_and_Clusters(G, data.num_nodes)
        # print('# Clusters with disjoint:  ' + str(disjoint_n_clusters_pred))
        # print(ID_pred)

        elapsed = time.time() - t
        # print('Elapsed prunning : ' + str(elapsed))
        num_CC = len(torch.unique(ID_pred))
        print('# CC = ' + str(num_CC))
    print('Done, computing metrics and mtmc')

    return ID_pred, predictions

def inference(CONFIG, loader, reid_model, mpn_model):

    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')

    reid_model.model.eval()
    mpn_model.eval()

    P_list = []
    R_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    TN_list = []
    F_list = []

    rand_index = []
    mutual_index = []
    homogeneity = []
    completeness = []
    v_measure = []

    precision_1_list = []
    precision_0_list = []

    tic = time.time()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= 0 :

                start_time = time.time()

                ########### Data extraction ###########
                bboxes = data

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
                        cam_ids_nodes.append(cams[c])

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
                                                             for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()

                # EDGE EMBEDDING
                node_dist_g = F.pairwise_distance(node_embeds_g[edge_ixs_g[0]], node_embeds_g[edge_ixs_g[1]]).view(-1,
                                                                                                                   1)
                node_dist_g_cos = 1 - F.cosine_similarity(node_embeds_g[edge_ixs_g[0]],
                                                          node_embeds_g[edge_ixs_g[1]]).view(
                    -1, 1)
                edge_attr = torch.cat((node_dist_g, node_dist_g_cos), dim=1)

                data = Data(x=node_embeds_g, edge_index=edge_ixs_g, y=node_labels_g, edge_attr=edge_attr,
                            edge_labels=edge_labels_g)

                if CONFIG['VISUALIZE']:
                    edge_labels_np = np.asarray(edge_labels_g.cpu())
                    node_labels_g_np = np.asarray(node_labels_g)

                    G = to_networkx(data, to_undirected=True)
                    utils.visualize(G, color=data.y, edge_labels=edge_labels_np, edge_index=edge_ixs_g_np,
                                    node_label=node_labels_g_np, mode='val')

                ########### Forward ###########

                outputs = mpn_model(data)
                labels_edges_GT = data.edge_labels.view(-1).cpu().numpy()

                preds = outputs['classified_edges'][-1]
                sof = torch.nn.Softmax(dim=1)
                preds_prob = sof(preds)

                predictions = torch.argmax(preds, dim=1)


                # CLUSTERING IDENTITIES MEASURES
                edge_list = data.edge_index.cpu().numpy()

                GT_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(labels_edges_GT) if p == 1]
                G_GT = nx.DiGraph(GT_active_edges)
                ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT,data.num_nodes)
                # print('# Clusters GT : ' + str(n_clusters_GT))
                # print(ID_GT)

                predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(predictions) if p == 1]
                G = nx.DiGraph(predicted_active_edges)
                ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G,data.num_nodes)
                # print('# Clusters:  ' + str(n_clusters_pred))
                # print(ID_pred)

                # POST PROCESSING
                num_cameras = len(loader.dataset.cameras)

                ID_pred, predictions = post_processing(num_cameras, ID_pred, predicted_active_edges, predictions, edge_list, CONFIG,
                                          data,
                                          preds_prob)

                ## CLUSTERING METRICS ##
                rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))

                TP, FP, TN, FN, P, R, FS, precision0, precision1 = compute_P_R_F(predictions, torch.from_numpy(labels_edges_GT).cuda())

                precision_1_list.append(np.sum(np.asarray([item for item in precision1])) / len(precision1))
                precision_0_list.append(np.sum(np.asarray([item for item in precision0])) / len(precision0))

                mutual_index.append(metrics.adjusted_mutual_info_score(ID_GT, ID_pred))
                homogeneity.append(metrics.homogeneity_score(ID_GT, ID_pred))
                completeness.append(metrics.completeness_score(ID_GT, ID_pred))
                v_measure.append(metrics.v_measure_score(ID_GT, ID_pred))


                TP_list.append(TP)
                FP_list.append(FP)
                FN_list.append(FN)
                P_list.append(P)
                R_list.append(R)
                F_list.append(FS)
                TN_list.append(TN)


                val_batch_time.update(time.time() - start_time)

                if i % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          '{et}<{eta}'.format(i, len(loader),
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(seconds=int(val_batch_time.avg * (len(loader) - i))))))


    #SAVE MTMC TRACKING RESULTS

    data_tracking = loader.dataset.data_det.copy()
    for n in range(len(nodes)):
        ID_new = int(ID_pred[n])
        CAM_ID = cam_ids_nodes[n]
        ID_old = node_labels_g[n]
        data_tracking.loc[(loader.dataset.data_det['id'] == ID_old).values & (
                loader.dataset.data_det['id_cam'] == CAM_ID).values, 'id'] = ID_new

    data_tracking = data_tracking[['id_cam', 'id', 'frame', 'xmin', 'ymin', 'width', 'height']]


    toc = time.time()

    print(['Time elapsed:  ' + str(toc-tic)])
    return P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index,mutual_index, homogeneity, completeness, v_measure, precision_0_list, precision_1_list, data_tracking

def inference_precomputed_features(CONFIG, loader, reid_model, mpn_model):
    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')

    reid_model.model.eval()
    mpn_model.eval()

    P_list = []
    R_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    TN_list = []
    F_list = []

    rand_index = []
    mutual_index = []
    homogeneity = []
    completeness = []
    v_measure = []

    precision_1_list = []
    precision_0_list = []

    tic = time.time()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= 0:

                start_time = time.time()

                ########### Data extraction ###########
                bboxes = data

                num_ids = len(bboxes)
                node_embeds_g = []
                node_labels_g = []
                cam_ids_nodes = []

                # COMPUTING NODE EMBEDDINGS,
                for n in range(num_ids): # only will be 1
                    id = bboxes[n]['id']
                    cams = bboxes[n]['cam']

                    for c in range(len(cams)): # only will be 1
                        feature = bboxes[n]['features'][0]
                        node_labels_g.append(id)
                        cam_ids_nodes.append(cams[c])

                        with torch.no_grad():
                            node_embed = (torch.from_numpy(feature)).cuda()
                            node_embeds_g.append(node_embed)

                node_embeds_g = torch.stack(node_embeds_g)
                if CONFIG['CNN_MODEL']['L2norm']:
                    node_embeds_g = F.normalize(node_embeds_g, p=2, dim=0)

                # COMPUTING EDGE INFORMATION
                edge_ixs_g = []
                nodes = np.asarray(range(len(cam_ids_nodes)))
                for id_cam in np.unique(cam_ids_nodes):
                    ids_in_cam = nodes[np.asarray(cam_ids_nodes) == id_cam]
                    ids_out_cam = nodes[np.asarray(cam_ids_nodes) != id_cam]
                    edge_ixs_g.append(torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))
                edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                edge_ixs_g_np = edge_ixs_g.cpu().numpy()
                # edges_temporal_iou = [(np.intersect1d(np.arange(bboxes[e[0]]['frames'][0], bboxes[e[0]]['frames'][1] + 1).reshape(-1, 1),
                #     np.arange(bboxes[e[1]]['frames'][0], bboxes[e[1]]['frames'][1] + 1).reshape(-1, 1))).shape[0] / (
                #            np.union1d(np.arange(bboxes[e[0]]['frames'][0], bboxes[e[0]]['frames'][1] + 1).reshape(-1, 1),
                #                np.arange(bboxes[e[1]]['frames'][0], bboxes[e[1]]['frames'][1] + 1).reshape(-1,1))).shape[0]   for e in edge_ixs_g_np.T]

                ########################## TEMPORAL THRESHOLDING ##########################
                # edges_temporal_exp = []
                # edges_temporal_dist = []
                # for e in edge_ixs_g_np.T:
                #     start1 = bboxes[e[0]]['frames'][0]
                #     start2 = bboxes[e[1]]['frames'][0]
                #     if start1 <= start2:
                #         # edges_temporal_exp.append(np.exp(-(start2-start1)/300))
                #         edges_temporal_dist.append(start2-start1)
                #     else:
                #         # edges_temporal_exp.append(np.exp(-(start1 - start2)/300))
                #         edges_temporal_dist.append(start1 - start2)
                #
                #
                # # x = np.array(np.arange(1, 100)).reshape(-1, 1)
                # # y = np.array(np.arange(101, 150)).reshape(-1, 1)
                # # intersec = np.intersect1d(x, y)
                # # union = np.union1d(x, y)
                # # iou = intersec.shape[0] / union.shape[0]
                # # edge_ixs_g_np = edge_ixs_g_np[:, np.asarray(edges_temporal_dist) < 1700]
                # edge_ixs_g = torch.from_numpy(edge_ixs_g_np).cuda()
                ########################## ##########################



                # # EDGE LABELS
                node_labels_g_np = np.asarray(node_labels_g)
                edge_labels_g = torch.from_numpy(np.asarray([1 if (node_labels_g_np[nodes == edge_ixs_g_np[0][i]] ==
                                                                   node_labels_g_np[
                                                                       nodes == edge_ixs_g_np[1][i]]) else 0
                                                             for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()

                # EDGE EMBEDDING
                node_dist_g = F.pairwise_distance(node_embeds_g[edge_ixs_g[0]], node_embeds_g[edge_ixs_g[1]]).view(-1,1)
                node_dist_g_cos = 1 - F.cosine_similarity(node_embeds_g[edge_ixs_g[0]],
                                                          node_embeds_g[edge_ixs_g[1]]).view(-1, 1)
                edge_attr = torch.cat((node_dist_g, node_dist_g_cos), dim=1)

                data = Data(x=node_embeds_g, edge_index=edge_ixs_g, y=node_labels_g, edge_attr=edge_attr, edge_labels=edge_labels_g)

                if CONFIG['VISUALIZE']:
                    edge_labels_np = np.asarray(edge_labels_g.cpu())
                    node_labels_g_np = np.asarray(node_labels_g)

                    G = to_networkx(data, to_undirected=True)
                    utils.visualize(G, color=data.y, edge_labels=edge_labels_np, edge_index=edge_ixs_g_np,    node_label=node_labels_g_np, mode='val')

                ########### Forward ###########
                tic = time.time()
                outputs, latent_node_feats  = mpn_model(data)

                print('Forward time = ' + str(time.time()-tic) + ' secs')
                if CONFIG['input_test'] == 'gt':
                    labels_edges_GT = data.edge_labels.view(-1)

                preds = outputs['classified_edges'][-1]
                sof = torch.nn.Softmax(dim=1)
                preds_prob = sof(preds)

                predictions = torch.argmax(preds, dim=1)

                # CLUSTERING IDENTITIES MEASURES
                edge_list = data.edge_index.cpu().numpy()


                predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos in torch.where(predictions == 1)[0]]


                G = nx.DiGraph(predicted_active_edges)
                ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G, data.num_nodes)
                # print('# Clusters:  ' + str(n_clusters_pred))
                # print(ID_pred)

                #POST PROCESSING

                num_cameras = len(loader.dataset.cameras)
                tic = time.time()
                ID_pred, predictions = post_processing(num_cameras, ID_pred, predicted_active_edges, predictions, edge_list, CONFIG, data,  preds_prob)
                print('Post-processing time = ' + str(time.time()-tic) + ' secs')


                if CONFIG['input_test'] == 'gt':

                    GT_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos in
                                       torch.where(labels_edges_GT == 1)[0]]

                    G_GT = nx.DiGraph(GT_active_edges)
                    ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT, data.num_nodes)
                ## CLUSTERING METRICS ##
                    rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))

                    TP, FP, TN, FN, P, R, FS, precision0, precision1 = compute_P_R_F(predictions, labels_edges_GT)

                    precision_1_list.append(np.sum(np.asarray([item for item in precision1])) / len(precision1))
                    precision_0_list.append(np.sum(np.asarray([item for item in precision0])) / len(precision0))

                    mutual_index.append(metrics.adjusted_mutual_info_score(ID_GT, ID_pred))
                    homogeneity.append(metrics.homogeneity_score(ID_GT, ID_pred))
                    completeness.append(metrics.completeness_score(ID_GT, ID_pred))
                    v_measure.append(metrics.v_measure_score(ID_GT, ID_pred))

                    TP_list.append(TP)
                    FP_list.append(FP)
                    FN_list.append(FN)
                    P_list.append(P)
                    R_list.append(R)
                    F_list.append(FS)
                    TN_list.append(TN)

                val_batch_time.update(time.time() - start_time)

                if i % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          '{et}<{eta}'.format(i, len(loader),
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(
                                                  seconds=int(val_batch_time.avg * (len(loader) - i))))))

    # SAVE MTMC TRACKING RESULTS

    data_tracking = loader.dataset.data_det.copy()
    for n in range(len(nodes)):
        ID_new = int(ID_pred[n])
        CAM_ID = cam_ids_nodes[n]
        ID_old = node_labels_g[n]
        # data_tracking['id'][(loader.dataset.data_det['id'] == ID_old).values & (
        #             loader.dataset.data_det['id_cam'] == CAM_ID).values] = ID_new

        data_tracking.loc[(loader.dataset.data_det['id'] == ID_old).values & (
                loader.dataset.data_det['id_cam'] == CAM_ID).values,'id'] = ID_new

    data_tracking = data_tracking[['id_cam', 'id', 'frame', 'xmin', 'ymin', 'width', 'height']]

    toc = time.time()

    print(['Time elapsed:  ' + str(toc - tic)])
    return P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index, mutual_index, homogeneity, completeness, v_measure, precision_0_list, precision_1_list, data_tracking
