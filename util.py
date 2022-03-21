
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

def get_reverse(_vector_distance_metric):
    if 'cos' in _vector_distance_metric or 'COS' in _vector_distance_metric:
        _reverse = True
    else:
        _reverse = False
    return _reverse


def get_recall_num(topk_max, _i2i_rm_self):
    if _i2i_rm_self:
        return 1, topk_max + 1
    else:
        return 0, topk_max


def cal_mean(list_in):
    if list_in is None or len(list_in) == 0:
        return 0.0
    else:
        return float(np.mean(list_in))


def add_metric_to_dict(name,
                       value,
                       metric_res,
                       _topk=None,
                       metric_size=None,
                       length_precision_recall_f=-1):
    if _topk is None:
        metric_res[name] = value
    else:
        if name not in metric_res:
            metric_res[name] = []
        metric_res[name].append({_topk: value})

    if length_precision_recall_f >= 0 and metric_size is not None:
        if name not in metric_size:
            metric_size[name] = []
        metric_size[name].append({_topk: length_precision_recall_f})


def write_metric(f_out, name, num, _topk):
    current_str = '{}@{}={}'.format(name, _topk, num)
    print('Info: ', current_str)
    f_out.write(current_str + '\n')


def write_metric_nok(f_out, name, num):
    current_str = '{}={}'.format(name, num)
    print('Info: ', current_str)
    f_out.write(current_str + '\n')


def cal_cos(arr1, arr2, _need_norm):
    """
    input
        arr1: query向量
        arr2: 多个目标item向量
    return:
        1x-array of cos scores
    """
    assert arr1.shape[-1] == arr2.shape[-1]
    num = np.matmul(arr1, arr2.T)
    if _need_norm:
        denom = (np.linalg.norm(arr1) *
                 np.linalg.norm(arr2, axis=1)).clip(min=1e-6)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
    else:
        sim = 0.5 + 0.5 * num
    if np.isnan(sim).any():
        print('Error: nan values in cal_cos!')
        print(arr1)
        print(arr2)
    return sim  # 转化为[0,1], 1: positive, 0: negative


def cal_l2(arr1, arr2):
    """
    input
        arr1: query向量
        arr2: 多个目标item向量
    return:
        1x-array of l2 scores
    """
    assert arr1.shape[-1] == arr2.shape[-1]
    res = np.linalg.norm(arr2 - arr1, axis=1)
    if np.isnan(res).any():
        print('Error: nan values! in cal_l2')
        print(arr1)
        print(arr2)
    return np.exp(-res)  # 转化为[0,1], 1: positive, 0: negative


def calcAUC(label, pred):
    if sum(label) == 0 or sum(label) == len(label):
        return np.nan
    try:
        fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=1)
        return metrics.auc(fpr, tpr)
    except Exception as e:
        print('Error: nan values in auc!')
        print(e)
        print(label)
        print(pred)
        return np.nan


def satisfy_score_limt(_vector_distance_metric, score_limit, current_score):
    if score_limit is None:
        return True
    if isinstance(current_score, str):
        current_score = float(current_score)
    if get_reverse(_vector_distance_metric):
        return current_score >= score_limit
    else:
        return current_score <= score_limit


def dcg_score(y_true, y_score, k=20, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true: array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score: array-like, shape = [n_samples]
        Predicted scores.
    k: int
        Rank.
    gains: str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k: float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=20, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true: array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score: array-like, shape = [n_samples]
        Predicted scores.
    k: int
        Rank.
    gains: str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k: float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


def list_all_files(path, hdfs=False):
    if hdfs:
        walk = tf.io.gfile.walk
    else:
        walk = os.walk

    list_r = []
    for root, dirs, files in walk(path, topdown=False):
        for name in files:
            list_r.append(os.path.join(root, name))

    return list_r


def add_current_entityids(args, query, item_entity_ids_dict):
    if args['if_i2i'] and query in item_entity_ids_dict:
        return [item_entity_ids_dict[query]]
    else:
        return []


def add_hdfs_prefix(path):
    if path.startswith('viewfs://hadoop-{your_prefix}'):
        return path
    else:
        return 'viewfs://hadoop-{your_prefix}' + path


def is_dir(path, hdfs=True):
    if hdfs:
        hdfs_path = add_hdfs_prefix(path)
        if tf.io.gfile.exists(hdfs_path):
            return tf.io.gfile.isdir(hdfs_path)
        else:
            raise Exception('{0} dose not exist in hdfs!'.format(hdfs_path))
    else:
        if os.path.exists(path):
            return os.path.isdir(path)
        else:
            raise Exception(
                '{0} dose not exist in local machine!'.format(path))


def get_files(path, _hdfs_or_local):
    if is_dir(path, _hdfs_or_local):
        if _hdfs_or_local:
            test_files = [
                add_hdfs_prefix(x)
                for x in list_all_files(add_hdfs_prefix(path), hdfs=True)
            ]
        else:
            test_files = list_all_files(path, hdfs=False)
    else:
        if _hdfs_or_local:
            test_files = [add_hdfs_prefix(path)]
        else:
            test_files = [path]

    return test_files


def get_file_and_size(eval_called_or_call_emb_name, _embs_hdfs_or_local):
    if _embs_hdfs_or_local:
        items_embs_in = tf.io.gfile.GFile(
            add_hdfs_prefix(eval_called_or_call_emb_name), 'r')
        item_file_size = items_embs_in.size()
    else:
        items_embs_in = codecs.open(eval_called_or_call_emb_name, 'r')
        item_file_size = os.path.getsize(eval_called_or_call_emb_name)
    return items_embs_in, item_file_size


def get_faiss_index(emb, _gpu_ids, _vector_distance_metric, _dim):
    print('Info: start to build index')
    embs = np.array(emb).astype(np.float32)  # inputs of faiss must be float32
    if _gpu_ids is None or _gpu_ids < 0:
        if 'cos' in _vector_distance_metric or 'COS' in _vector_distance_metric:
            index = faiss.IndexFlatIP(_dim)
        else:
            index = faiss.IndexFlatL2(_dim)
        index.add(embs)
        return index
    else:
        try:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = False
            flat_config.device = _gpu_ids
            if 'cos' in _vector_distance_metric or 'COS' in _vector_distance_metric:
                gpu_index = faiss.GpuIndexFlatIP(res, _dim, flat_config)
            else:
                gpu_index = faiss.GpuIndexFlatL2(res, _dim, flat_config)
            gpu_index.add(embs)

        except Exception as e:
            print('Error: ', e)
            print(
                'Warning: because of the upper problem of gpu, now use cpu to do faiss-search'
            )
            if 'cos' in _vector_distance_metric or 'COS' in _vector_distance_metric:
                gpu_index = faiss.IndexFlatIP(_dim)
            else:
                gpu_index = faiss.IndexFlatL2(_dim)
            gpu_index.add(embs)
        return gpu_index



def str2bool(v):
    if v.lower() in ('yes', 'Yes', 'true', 'True', '1'):
        return True
    elif v.lower() in ('no', 'No', 'false', 'False', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def report_result(callback_url, task_id, result):
    """
    callback_url: url
    task_id: id
    result: result dict
    """
    print('callback_url:', callback_url, 'task_id:', task_id)
    if callback_url is None or len(
            callback_url) == 0 or task_id is None or len(task_id) == 0:
        print('callback_url or task_id is empty, report cancelled.')
        return

    data = {
        "taskId": task_id,
        "status": 3,
        "data": [{
            "name": k,
            "value": v
        } for k, v in result.items()]
    }

    for i in range(10):
        try:
            response = requests.post(callback_url, json=data)
            if response.status_code == 200:
                break
        except Exception as e:
            logging.exception("report result error.", e)
        time.sleep(5)

    print('report done, status:', response.status_code)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    else:
        return text
