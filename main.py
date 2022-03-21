# -*- coding:utf-8 -*-
"""
# @time    : 2020/9/23 下午8:32
# @author  : zhixiaochuan12
# @desc    : precision@k recall@k f-measure@k hitrate@k map@k mrr@k ndcg@k auc qauc qrecall qndcg
"""

import argparse
import codecs
import math
import faiss
import numpy as np
import os
import json
import tensorflow as tf
from tqdm import tqdm
import logging
import sys
import time
from sklearn import metrics
import operator
import base64
import requests
import time

from util import *

global_show_info_cnt = 0


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='metric calculation')

    # required params
    parser.add_argument('--topk', type=str, default='50', help='metric@k')
    parser.add_argument(
        '--topm',
        type=str,
        default='',
        help='metric@topm for each call_node in sort_and_merge way')
    parser.add_argument('--metric',
                        type=str,
                        default='hitrate',
                        help='metrics')
    parser.add_argument('--sep',
                        type=str,
                        default='#',
                        help='sep mark between items')
    parser.add_argument('--vector_distance_metric',
                        type=str,
                        default='cos',
                        help='l2 or cos')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='if gpu_ids>=0 use gpu, otherwise cpu')
    parser.add_argument('--beta',
                        type=float,
                        default=1.0,
                        help='f_beta, used in f_measure')
    parser.add_argument('--eval_call_emb_name',
                        type=str,
                        default='',
                        help='eval_call_emb_name')
    parser.add_argument('--eval_called_emb_name',
                        type=str,
                        default='',
                        help='eval_called_emb_name')
    parser.add_argument('--test_seqs_name',
                        type=str,
                        default='',
                        help='test_seqs_name')

    parser.add_argument('--embs_hdfs_or_local',
                        type=str2bool,
                        default=True,
                        help='hdfs or local')
    parser.add_argument('--test_seqs_hdfs_or_local',
                        type=str2bool,
                        default=True,
                        help='hdfs or local')
    parser.add_argument(
        '--embs_or_topk',
        type=str2bool,
        default=True,
        help='embs_or_topk, false,topk; topk_format: item:score;item:score...')
    parser.add_argument('--if_i2i',
                        type=str2bool,
                        default=False,
                        help='if_i2i, true or false')
    parser.add_argument('--batch_size',
                        type=int,
                        default=10240,
                        help='batch_size to eval...')
    parser.add_argument('--emb_sep', type=str, default=' ', help='emb_sep')
    parser.add_argument(
        '--call_node_merge_way',
        type=str,
        default='last',
        help=
        'merge way of multi call nodes embs and nearst results, including <last/mean_pooling/max_pooling/sort_and_merge>'
    )
    parser.add_argument('--sort_and_merge_last_k',
                        type=int,
                        default=-1,
                        help='default all')
    parser.add_argument('--score_limit',
                        type=float,
                        default=None,
                        help='score_limit for global metrics')
    parser.add_argument('--call_node_keep_prefix',
                        type=str,
                        default='',
                        help='')
    parser.add_argument('--called_node_keep_prefix',
                        type=str,
                        default='',
                        help='')
    parser.add_argument('--call_node_rm_prefix', type=str, default='', help='')
    parser.add_argument('--called_node_rm_prefix',
                        type=str,
                        default='',
                        help='')
    parser.add_argument('--call_node_add_prefix',
                        type=str,
                        default='',
                        help='add prefix to entities in call node embs')
    parser.add_argument('--called_node_add_prefix',
                        type=str,
                        default='',
                        help='add prefix to entities in called node embs')
    parser.add_argument('--need_norm',
                        type=str2bool,
                        default=False,
                        help='if need norm')
    parser.add_argument('--i2i_rm_self',
                        type=str2bool,
                        default=False,
                        help='if _i2i_rm_self')
    parser.add_argument(
        '--per_user_avg',
        type=str2bool,
        default=False,
        help=
        'if use per-user average instead of global average, used in precision/recall/f_measure'
    )
    parser.add_argument('--eval_globalid_idx', type=int, default=0, help='')
    parser.add_argument('--eval_call_nodesession_mapping_idx',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--eval_called_nodesession_mapping_idx',
                        type=int,
                        default=2,
                        help='')
    parser.add_argument('--labels_mapping_idx', type=int, default=3, help='')

    args = parser.parse_args(argv)
    print('Info: args: ', args)
    return args

def read_item_embs_or_topk(args):
    item_ids = []
    item_embeddings = []
    item_embeddings_or_topkinfo = {}
    item_embs_files = get_files(args['eval_called_emb_name'],
                                args['embs_hdfs_or_local'])
    with tqdm(len(item_embs_files), total=len(item_embs_files)) as pbar_out:
        for i in range(len(item_embs_files)):
            item_embs_file = item_embs_files[i]
            pbar_out.set_description('reading item embeddings')
            pbar_out.update(1)
            items_embs_in, item_file_size = get_file_and_size(
                item_embs_file, args['embs_hdfs_or_local'])
            with tqdm(item_file_size, total=item_file_size) as pbar:
                pbar.set_description(
                    'reading item embeddings {0}/{1}...'.format(
                        i, len(item_embs_files)))
                for line in items_embs_in:
                    pbar.update(len(line))
                    splits = line.strip().split('\t')
                    if len(args['called_node_keep_prefix']) > 0 and (
                            not splits[0].startswith(
                                args['called_node_keep_prefix'])):
                        continue

                    def norm_entity(x):
                        return args['called_node_add_prefix'] + remove_prefix(
                            x, args['called_node_rm_prefix'])

                    current_item_entity = norm_entity(splits[0])

                    item_ids.append(current_item_entity)
                    if args['embs_or_topk']:
                        vector = np.array([
                            float(x)
                            for x in splits[1].strip().split(args['emb_sep'])
                        ],
                            dtype=np.float32)
                        if args['dim'] < 0:
                            args['dim'] = len(vector)
                            print('Info: vector dim: {0}'.format(args['dim']))
                        if args['need_norm']:
                            norm = np.linalg.norm(vector)
                            if norm > 0:
                                vector = vector / norm
                        assert len(vector) == args['dim']
                        item_embeddings.append(vector)
                    else:
                        tops = splits[1].split(';')
                        item_scores_tuple = [x.split(':') for x in tops]
                        item_scores = [
                            x for x in item_scores_tuple if satisfy_score_limt(
                                args['vector_distance_metric'],
                                args['score_limit'], x[1])
                        ]
                        items = [
                            norm_entity(x[0]) for x in item_scores
                            if (not args['i2i_rm_self']) or (
                                    args['i2i_rm_self']
                                    and norm_entity(x[0]) != current_item_entity)
                        ]
                        scores = [float(x[1]) for x in item_scores]
                        vector = (items, scores)
                    item_embeddings_or_topkinfo[current_item_entity] = vector
    return item_ids, item_embeddings, item_embeddings_or_topkinfo, args['dim']


def get_query_embs_or_topk(args):
    query_embeddings = {}
    query_embs_files = get_files(args['eval_call_emb_name'],
                                 args['embs_hdfs_or_local'])
    with tqdm(len(query_embs_files), total=len(query_embs_files)) as pbar_out:
        for i in range(len(query_embs_files)):
            query_embs_file = query_embs_files[i]
            pbar_out.set_description(
                'reading subject(user/query/item) embeddings')
            pbar_out.update(1)
            query_embs_in, query_file_size = get_file_and_size(
                query_embs_file, args['embs_hdfs_or_local'])
            with tqdm(query_file_size, total=query_file_size) as pbar:
                pbar.set_description(
                    'reading subject(user/query/item) embeddings {0}/{1}'.
                        format(i, len(query_embs_files)))
                for line in query_embs_in:
                    pbar.update(len(line))
                    splits = line.strip('\n').split('\t')
                    if len(args['call_node_keep_prefix']) > 0 and (
                            not splits[0].startswith(
                                args['call_node_keep_prefix'])):
                        continue
                    current_query_entity = args[
                                               'call_node_add_prefix'] + remove_prefix(
                        splits[0], args['call_node_rm_prefix'])
                    vector = np.array([
                        float(x)
                        for x in splits[1].strip().split(args['emb_sep'])
                    ],
                        dtype=np.float32)
                    if args['need_norm']:
                        norm = np.linalg.norm(vector)
                        if norm > 0:
                            vector = vector / norm
                    assert len(vector) == args['dim']
                    query_embeddings[current_query_entity] = np.array(
                        vector, dtype=np.float32)
    return query_embeddings


def process_batchs(args, user_vectors, globalids, globalid_clicked_items_dict,
                   index, item_embeddings_or_topkinfo, item_ids,
                   current_entity_ids, total_ndcg, mrr, map, precisions,
                   recalls, hit_sum_pr, recall_count, precision_count,
                   hitrate_sum):
    if len(user_vectors) == 0:
        return
    _topk_max = max(args['topms'])
    global global_show_info_cnt
    if args['embs_or_topk']:
        user_vectors = np.array(user_vectors, dtype=np.float32)
        if len(user_vectors.shape) == 1:
            user_vectors = user_vectors.reshape([1, -1])
        try:
            distss, idss = index.search(x=user_vectors,
                                        k=get_recall_num(
                                            _topk_max, args['i2i_rm_self'])[1])
            distss = distss.tolist()
            idss = idss.tolist()
            dists = []
            ids = []
            for dists_one, ids_one, current_entity_id_one in zip(
                    distss, idss, current_entity_ids):
                dists_ids_one = [
                    x for x in zip(dists_one, ids_one)
                    if satisfy_score_limt(args['vector_distance_metric'],
                                          args['score_limit'], x[0])
                ]
                dists.append([
                    x[0] for x in dists_ids_one if (not args['i2i_rm_self']) or
                                                   (args['i2i_rm_self'] and x[1] not in current_entity_id_one)
                ])
                ids.append([
                    x[1] for x in dists_ids_one if (not args['i2i_rm_self']) or
                                                   (args['i2i_rm_self'] and x[1] not in current_entity_id_one)
                ])
        except Exception as e:
            print('Error:', e)
            print(user_vectors.shape)
            print(user_vectors)
            return
    else:
        if global_show_info_cnt < 5:
            print('Info: call items top5, ', user_vectors[0:5])
        ids = [
            item_embeddings_or_topkinfo[x][0][0:_topk_max]
            for x in user_vectors
        ]
        dists = [
            item_embeddings_or_topkinfo[x][1][0:_topk_max]
            for x in user_vectors
        ]

    globalid_dist_ids = {}
    if args['call_node_merge_way'] == 'sort_and_merge':
        for globalid, dist, id in zip(globalids, dists, ids):
            if globalid not in globalid_dist_ids:
                globalid_dist_ids[globalid] = []
            globalid_dist_ids[globalid].append((dist, id))
        for globalid, dist_ids in globalid_dist_ids.items():
            if len(dist_ids) != 1:
                dists = []
                ids = []
                for dist_id in dist_ids:
                    assert len(dist_id[0]) == len(dist_id[1])
                    dists.extend([float(x) for x in dist_id[0]])
                    ids.extend([x for x in dist_id[1]])
                dists_merge_sorted, ids_merge_sorted = zip(*sorted(
                    zip(dists, ids),
                    key=operator.itemgetter(0),
                    reverse=get_reverse(args['vector_distance_metric'])))
                globalid_dist_ids[globalid] = (dists_merge_sorted,
                                               ids_merge_sorted)
            else:
                globalid_dist_ids[globalid] = dist_ids[0]
    else:
        for globalid, dist, id in zip(globalids, dists, ids):
            globalid_dist_ids[globalid] = (dist, id)
    for globalid, dist_ids in globalid_dist_ids.items():

        clicked_items_set_i = globalid_clicked_items_dict[globalid]
        for _topk in args['topks']:
            if args['embs_or_topk']:
                embs_idx_start, embs_idx_end = get_recall_num(
                    _topk, args['i2i_rm_self'])
                predicted_items = [
                    item_ids[id]
                    for id in dist_ids[1][embs_idx_start:embs_idx_end]
                ]
            else:
                predicted_items = [id for id in dist_ids[1][:_topk]]
            predicted_items_set = set(predicted_items)
            if global_show_info_cnt < 5 and _topk == args['topks'][-1]:
                print(
                    'Info: Sampled {}th predicted_items_set_i:'.format(
                        global_show_info_cnt), predicted_items_set)
                print(
                    'Info: Sampled {}th clicked_items_set_i:'.format(
                        global_show_info_cnt), clicked_items_set_i)
                global_show_info_cnt += 1

            # ndcg
            if args['ndcg']:
                recall_ndcg = 0
                dcg = 0.0
                for num, predicted_item in enumerate(predicted_items):
                    if predicted_item in clicked_items_set_i:
                        recall_ndcg += 1
                        dcg += 1.0 / math.log(num + 2, 2)
                idcg = 0.0
                for num in range(recall_ndcg):
                    idcg += 1.0 / math.log(num + 2, 2)
                if recall_ndcg > 0:
                    total_ndcg[_topk].append(dcg / idcg)
                else:
                    total_ndcg[_topk].append(0.0)

            # mrr
            if args['mrr']:
                for clicked_item in clicked_items_set_i:
                    if clicked_item in predicted_items:
                        idx = predicted_items.index(clicked_item)
                        mrr[_topk].append(1.0 / (idx + 1))
                    else:
                        mrr[_topk].append(0.0)

            # map
            if args['map']:
                ap = 0.0
                hit_map = 0
                for idx, predicted_item in enumerate(predicted_items):
                    if predicted_item in clicked_items_set_i:
                        hit_map += 1
                        ap += hit_map * 1.0 / (idx + 1)
                map[_topk].append(ap / len(clicked_items_set_i))

            # precision recall f_measure
            if args['precision'] or args['recall'] or args['f_measure']:
                hit_one = len([
                    x for x in clicked_items_set_i if x in predicted_items_set
                ])
                if args['per_user_avg']:
                    if len(predicted_items_set) > 0:
                        precisions[_topk].append(hit_one * 1.0 /
                                                 len(predicted_items_set))
                    recalls[_topk].append(hit_one * 1.0 /
                                          len(clicked_items_set_i))
                else:
                    hit_sum_pr[_topk] += hit_one
                    recall_count[_topk] += len(clicked_items_set_i)
                    precision_count[_topk] += len(predicted_items_set)

            # hitrate
            if args['hitrate']:
                hitrate_sum[_topk].append(
                    int(len(clicked_items_set_i & predicted_items_set) > 0))


def main(argv):
    args = get_args(argv)
    args = vars(args)
    _begin_time = time.time()
    args['dim'] = -1
    assert len(args['eval_call_emb_name']) > 0
    assert len(args['eval_called_emb_name']) > 0
    assert len(args['test_seqs_name']) > 0

    print('Info: args (after reading json config): ', args)

    print('Info: eval_call_emb_name: ', args['eval_call_emb_name'])
    print('Info: eval_called_emb_name: ', args['eval_called_emb_name'])
    print('Info: eval_test_seqs_name: ', args['test_seqs_name'])
    '''verify'''

    args['topks'] = [int(x) for x in args['topk'].split(',')]
    if args['topm'] and len(args['topm']) > 0:
        args['topms'] = [int(x) for x in args['topm'].split(',')]
    else:
        args['topms'] = args['topks']
    if len(args['topks']) != len(args['topms']):
        raise Exception('topks should has the same size with topms')

    assert args['call_node_merge_way'] in [
        'last', 'mean_pooling', 'max_pooling', 'sort_and_merge'
    ]
    args['metrics'] = args['metric'].split(',')
    args['recall'] = True if 'recall' in args['metrics'] else False
    args['precision'] = True if 'precision' in args['metrics'] else False
    args['f_measure'] = True if 'f_measure' in args['metrics'] else False
    args['hitrate'] = True if 'hitrate' in args['metrics'] else False
    args['map'] = True if 'map' in args['metrics'] else False
    args['mrr'] = True if 'mrr' in args['metrics'] else False
    args['ndcg'] = True if 'ndcg' in args['metrics'] else False
    args['auc'] = True if 'auc' in args['metrics'] else False
    args['qauc'] = True if 'qauc' in args['metrics'] else False
    args['qndcg'] = True if 'qndcg' in args['metrics'] else False
    args['qrecall'] = True if 'qrecall' in args['metrics'] else False
    assert args['recall'] or args['precision'] or args['f_measure'] or args['map'] or args['mrr'] or args['ndcg'] or \
           args['hitrate'] or args['auc'] or args['qauc'] or args['qrecall'] or args['qndcg']

    item_ids, item_embeddings, item_embeddings_or_topkinfo, args[
        'dim'] = read_item_embs_or_topk(args)
    item_entity_ids_dict = {k: id for id, k in enumerate(item_ids)}

    if args['call_node_merge_way'] == 'sort_and_merge' and (
            not args['embs_or_topk']):
        assert args[
            'if_i2i'], 'call_node_merge_way=sort_and_merge only supports 1.embs and 2.topk and i2i.'
    if not args['embs_or_topk']:
        assert args[
                   'call_node_merge_way'] == 'sort_and_merge', 'if topk, only supports call_node_merge_way=sort_and_merge'
    if args['if_i2i']:
        query_embeddings = item_embeddings_or_topkinfo
    else:
        query_embeddings = get_query_embs_or_topk(args)
    if 'i2i_rm_self':
        assert args['call_node_merge_way'] not in (
            'max_pooling', 'mean_pooling'
        ), "i2i_rm_self not support max_pooling or mean_pooling, for there are multi nodes to call."
    print("Info: eval_call_emb_size={}".format(len(query_embeddings)))
    print("Info: eval_called_emb_size={}".format(
        len(item_embeddings_or_topkinfo)))

    print('Info: item_embeddings_or_topkinfo head:')
    print(dict(list(item_embeddings_or_topkinfo.items())[0:5]))
    print('Info: subject(user/query/item) embeddings head:')
    print(dict(list(query_embeddings.items())[0:5]))

    recall_count = {}
    precision_count = {}
    hit_sum_pr = {}
    recalls = {}
    precisions = {}
    hitrate_sum = {}
    map = {}
    mrr = {}
    total_ndcg = {}
    recall_sum = {}
    q_total_ndcg = {}
    for _topk in args['topks']:
        recalls[_topk] = []
        precisions[_topk] = []
        hitrate_sum[_topk] = []
        map[_topk] = []
        mrr[_topk] = []
        total_ndcg[_topk] = []
        recall_sum[_topk] = []
        recall_count[_topk] = 0
        precision_count[_topk] = 0
        hit_sum_pr[_topk] = 0
        q_total_ndcg[_topk] = []

    labels_auc = []
    scores_auc = []
    sum_qauc = []
    index = None
    if args['embs_or_topk']:
        index = get_faiss_index(item_embeddings, args['gpu'],
                                args['vector_distance_metric'], args['dim'])
        print('Info: index has been built')

    test_files = get_files(args['test_seqs_name'],
                           args['test_seqs_hdfs_or_local'])
    line_call_absent_cnt = 0
    line_called_absent_cnt = 0
    line_exist_cnt = 0
    line_cnt = 0
    with tqdm(range(len(test_files)), total=len(test_files)) as pbar:
        for ii in range(len(test_files)):
            pbar.set_description('正在评测第{0}/{1}个文件'.format(
                ii + 1, len(test_files)))
            pbar.update(1)
            test_file = test_files[ii]
            file_in = tf.io.gfile.GFile(test_file, 'r')

            with tqdm(file_in.size(), total=file_in.size()) as pbar0:
                pbar0.set_description('第{0}/{1}个文件处理进度'.format(
                    ii + 1, len(test_files)))
                user_vectors = []
                current_entity_ids = []
                globalids = []
                globalid_clicked_items_dict = {}
                for line in file_in:
                    line_cnt += 1
                    pbar0.update(len(line))
                    line = line.strip('\n')
                    splits = line.split('\t')
                    globalid = splits[args['eval_globalid_idx']]
                    globalid = str(globalid) + '_' + str(
                        line_cnt)  # keep uniqueness
                    queries = [
                        x for x in splits[
                            args['eval_call_nodesession_mapping_idx']].split(
                            args['sep']) if x in query_embeddings
                    ]
                    exposed_items_raw = [
                        x for x in splits[
                            args['eval_called_nodesession_mapping_idx']].split(
                            args['sep'])
                    ]
                    exposed_items_labels_raw = [
                        int(y) for x, y in zip(
                            splits[args['eval_called_nodesession_mapping_idx']]
                                .split(args['sep']), splits[
                                args['labels_mapping_idx']].split(args['sep']))
                    ]
                    exposed_items = [
                        x for x in splits[
                            args['eval_called_nodesession_mapping_idx']].split(
                            args['sep'])
                        if x in item_embeddings_or_topkinfo
                    ]
                    exposed_items_labels = [
                        int(y) for x, y in zip(
                            splits[args['eval_called_nodesession_mapping_idx']]
                                .split(args['sep']), splits[
                                args['labels_mapping_idx']].split(args['sep']))
                        if x in item_embeddings_or_topkinfo
                    ]
                    if len(exposed_items) == 0 or len(queries) == 0:
                        if line_call_absent_cnt <= 5:
                            print('Info: the ', line_call_absent_cnt,
                                  'th absent line:', globalid, splits, queries,
                                  exposed_items)
                        line_call_absent_cnt += 1
                        continue

                    clicked_items_raw = [
                        x for x, lb in zip(exposed_items_raw,
                                           exposed_items_labels_raw) if lb == 1
                    ]
                    clicked_items_raw_set = set(clicked_items_raw)

                    if len(clicked_items_raw_set) == 0:
                        line_called_absent_cnt += 1
                        continue
                    globalid_clicked_items_dict[
                        globalid] = clicked_items_raw_set
                    if line_exist_cnt <= 5:
                        print('Info: the ', line_exist_cnt,
                              'th existing line:', globalid, splits, queries,
                              exposed_items)
                    line_exist_cnt += 1
                    if args['call_node_merge_way'] == 'sort_and_merge':  ## for i2i
                        sort_and_merge_last_k = len(
                            queries
                        ) if args['sort_and_merge_last_k'] < 0 else args[
                            'sort_and_merge_last_k']
                        for query in queries[-sort_and_merge_last_k:]:
                            if args['if_i2i'] and (
                                    not args['embs_or_topk']
                            ) and query in item_embeddings_or_topkinfo:
                                user_vectors.append(query)
                                current_entity_ids.append(
                                    add_current_entityids(
                                        args, query, item_entity_ids_dict))
                                globalids.append(globalid)
                            else:
                                if query in query_embeddings and args[
                                    'embs_or_topk']:
                                    current_entity_ids.append(
                                        add_current_entityids(
                                            args, query, item_entity_ids_dict))
                                    globalids.append(globalid)

                    else:
                        if args['call_node_merge_way'] == 'last':
                            current_user_vectors = query_embeddings[
                                queries[-1]]
                            current_entity_ids.append(
                                add_current_entityids(args, queries[-1],
                                                      item_entity_ids_dict))
                        elif args['call_node_merge_way'] == 'mean_pooling':
                            current_user_vectors = np.mean([
                                query_embeddings[queries[x]]
                                for x in range(len(queries))
                            ],
                                axis=0)
                            current_entity_ids.append([])
                        elif args['call_node_merge_way'] == 'max_pooling':
                            current_user_vectors = np.max([
                                query_embeddings[queries[x]]
                                for x in range(len(queries))
                            ],
                                axis=0)
                            current_entity_ids.append([])
                        else:
                            raise Exception(
                                'call_node_merge_way has been set wrong!')

                        user_vectors.append(
                            current_user_vectors)  # 存储vectors或者topk call terms
                        globalids.append(globalid)

                        if (args['auc'] or args['qauc'] or args['qrecall']
                            or args['qndcg']) and args['embs_or_topk']:
                            if 'cos' in args[
                                'vector_distance_metric'] or 'COS' in args[
                                'vector_distance_metric']:
                                predicted_item_scores = cal_cos(
                                    current_user_vectors,
                                    np.array([
                                        item_embeddings_or_topkinfo[x]
                                        for x in exposed_items
                                    ],
                                        dtype=np.float32),
                                    args['need_norm'])
                            else:
                                predicted_item_scores = cal_l2(
                                    current_user_vectors,
                                    np.array([
                                        item_embeddings_or_topkinfo[x]
                                        for x in exposed_items
                                    ],
                                        dtype=np.float32))

                            assert len(exposed_items_labels) == len(
                                exposed_items) == len(predicted_item_scores)
                            if args['auc']:
                                labels_auc.extend(exposed_items_labels)
                                scores_auc.extend(predicted_item_scores)
                            if args['qauc'] or args['qrecall'] or args['qndcg']:
                                sort_idxes = np.array(
                                    predicted_item_scores).argsort()[::-1]
                                hit_sum = {}
                                for _topk in args['topks']:
                                    hit_sum[_topk] = 0
                                for i, sort_idx in enumerate(sort_idxes):
                                    if exposed_items_labels[sort_idx] == 1:
                                        for _topk in args['topks']:
                                            if i < _topk:
                                                hit_sum[_topk] += 1
                                for _topk in args['topks']:
                                    recall_sum[_topk].append(
                                        hit_sum[_topk] * 1.0 /
                                        len(clicked_items_raw)
                                    )  # clicked_items_set may cause qrecall>1
                                    qndcg = ndcg_score(exposed_items_labels,
                                                       predicted_item_scores,
                                                       k=_topk)
                                    if not np.isnan(qndcg):
                                        q_total_ndcg[_topk].append(qndcg)
                                auc = calcAUC(exposed_items_labels,
                                              predicted_item_scores)
                                if not np.isnan(auc):
                                    sum_qauc.append(auc)
                    if len(user_vectors) > args['batch_size']:
                        process_batchs(args, user_vectors, globalids,
                                       globalid_clicked_items_dict, index,
                                       item_embeddings_or_topkinfo, item_ids,
                                       current_entity_ids, total_ndcg, mrr,
                                       map, precisions, recalls, hit_sum_pr,
                                       recall_count, precision_count,
                                       hitrate_sum)
                        user_vectors = []
                        globalids = []
                        globalid_clicked_items_dict = {}

                process_batchs(args, user_vectors, globalids,
                               globalid_clicked_items_dict, index,
                               item_embeddings_or_topkinfo, item_ids,
                               current_entity_ids, total_ndcg, mrr, map,
                               precisions, recalls, hit_sum_pr, recall_count,
                               precision_count, hitrate_sum)

            file_in.close()
        print('Info: num of all lines:', line_cnt)
        print('Info: num of evaluated lines:', line_exist_cnt)
        print(
            'Info: num of missed lines (call nodes not exist in call_embs.):',
            line_call_absent_cnt)
        print(
            'Info: num of missed lines (called nodes not exist in called_embs or no click nodes.):',
            line_called_absent_cnt)

    f_out = codecs.open('eval_log_{0}.txt'.format(_begin_time), 'w')
    f_out.write(str(args) + '\n\n')
    if (args['auc'] or args['qauc'] or args['qrecall']
        or args['qndcg']) and not args['embs_or_topk']:
        tip_info = 'Warning: when topk info provided instead of embs, auc/qauc/qrecall/qndcg metrics can not be calculated.\n'
        print(tip_info)
        f_out.write(tip_info + '\n')
    metric_res = {}
    metric_size = {}
    for _topk in args['topks']:
        length_precision_recall_f = -1
        if args['precision'] or args['recall'] or args['f_measure']:
            if args['per_user_avg']:
                final_presicion = cal_mean(precisions[_topk])
                final_recall = cal_mean(recalls[_topk])
                length_precision_recall_f = len(precisions[_topk])
            else:
                final_presicion = hit_sum_pr[_topk] * 1.0 / precision_count[
                    _topk]
                final_recall = hit_sum_pr[_topk] * 1.0 / recall_count[_topk]
                length_precision_recall_f = recall_count[_topk]
            if args['precision']:
                add_metric_to_dict('precision', final_presicion, metric_res,
                                   _topk, metric_size,
                                   length_precision_recall_f)
            if args['recall']:
                add_metric_to_dict('recall', final_recall, metric_res, _topk,
                                   metric_size, length_precision_recall_f)
            if args['f_measure']:
                if final_presicion > 0 and final_presicion > 0:
                    f_value = (1 + math.pow(
                        args['beta'], 2)) * final_presicion * final_recall / (
                                      math.pow(args['beta'], 2) * final_presicion +
                                      final_recall)
                    add_metric_to_dict('f_{}'.format(args['beta']), f_value,
                                       metric_res, _topk, metric_size,
                                       length_precision_recall_f)
                else:
                    add_metric_to_dict('f_{}'.format(args['beta']), 0,
                                       metric_res, _topk, metric_size,
                                       length_precision_recall_f)
        if args['hitrate']:
            hitrate_value = cal_mean(hitrate_sum[_topk])
            add_metric_to_dict('hitrate', hitrate_value, metric_res, _topk,
                               metric_size, len(hitrate_sum[_topk]))
        if args['map']:
            map_value = cal_mean(map[_topk])
            add_metric_to_dict('map', map_value, metric_res, _topk,
                               metric_size, len(map[_topk]))
        if args['mrr']:
            mrr_value = cal_mean(mrr[_topk])
            add_metric_to_dict('mrr', mrr_value, metric_res, _topk,
                               metric_size, len(mrr[_topk]))
        if args['ndcg']:
            ndcg_value = cal_mean(total_ndcg[_topk])
            add_metric_to_dict('ndcg', ndcg_value, metric_res, _topk,
                               metric_size, len(total_ndcg[_topk]))
        if args['qrecall'] and args['embs_or_topk']:
            qrecall_value = cal_mean(recall_sum[_topk])
            add_metric_to_dict('qrecall', qrecall_value, metric_res, _topk,
                               metric_size, len(recall_sum[_topk]))
        if args['qndcg'] and args['embs_or_topk']:
            qndcg_value = cal_mean(q_total_ndcg[_topk])
            add_metric_to_dict('qndcg', qndcg_value, metric_res, _topk,
                               metric_size, len(q_total_ndcg[_topk]))
    if args['auc'] and args['embs_or_topk']:
        auc_value = calcAUC(labels_auc, scores_auc)
        add_metric_to_dict('auc', auc_value, metric_res, _topk, metric_size,
                           len(labels_auc))
    if args['qauc'] and args['embs_or_topk']:
        qauc_value = cal_mean(sum_qauc)
        add_metric_to_dict('qauc', qauc_value, metric_res, _topk, metric_size,
                           len(sum_qauc))
    time_cost = time.time() - _begin_time
    print('Info: time cost for evaluation: {0} s'.format("%.2f" % time_cost))
    print('Info: metrics result = {}'.format(metric_res))
    print('Info: metric size = {}'.format(metric_size))

    f_out.close()


if __name__ == '__main__':
    main(sys.argv[1:])
