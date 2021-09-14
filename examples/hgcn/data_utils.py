# Copyright (C) 2016-2021 Alibaba Group Holding Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
sys.path.append('.')

import numpy as np
import pickle as pkl
import tensorflow as tf

import scipy.sparse as sp

from config import training_config


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def augment(adj, features):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = np.array(np.eye(6)[deg], dtype=np.float32).squeeze()
    const_f = np.ones((features.shape[0], 1))
    features = np.concatenate([features, deg_onehot, const_f], axis=1)
    return features


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])
    if normalize_adj:
        adj = normalize(adj)
    adj = np.array(adj.toarray(), dtype=np.float32)
    return adj, features


def mask_edges(adj, val_prop):
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)
    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    val_edges, train_edges = pos_edges[-n_val:], pos_edges[:-n_val]
    val_edges_false = neg_edges[-n_val:]
    train_edges_false = np.concatenate([neg_edges, val_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false


def load_data_airport(data_path, normalize_adj, normalize_feat):
    adj = pkl.load(open(os.path.join(data_path, "adj.pkl"), 'rb'))
    features = pkl.load(open(os.path.join(data_path, "features.pkl"), 'rb'))
    adj = sp.csr_matrix(adj)
    data = {'adj_train': adj, 'features': features}
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false = mask_edges(adj, 0.2)
    data['adj_train'] = adj_train
    data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
    data['val_edges'] = [list(p) + [1] for p in val_edges] + [list(p) + [0] for p in val_edges_false]
    data['adj_train_norm'], data['features'] = process(data['adj_train'], data['features'], normalize_adj,
                                                       normalize_feat)
    data['features'] = np.array(augment(data['adj_train'], data['features']), dtype=np.float32)
    return data


def get_train_batch(pos_edge_set, neg_edge_set):
    pos_dataset = tf.data.Dataset.from_tensor_slices(pos_edge_set)
    pos_dataset = pos_dataset.shuffle(buffer_size=training_config["shuffle_buffer"]).batch(
        training_config["batch_size"], drop_remainder=False)
    pos_dataset = pos_dataset.repeat()
    pos_iter = pos_dataset.make_one_shot_iterator()
    pos_batch = pos_iter.get_next()
    pos_label = tf.ones(shape=(tf.shape(pos_batch)[0]))
    neg_dataset = tf.data.Dataset.from_tensor_slices(neg_edge_set)
    neg_dataset = neg_dataset.shuffle(buffer_size=training_config["shuffle_buffer"]).batch(
        training_config["neg_samples"] * training_config["batch_size"], drop_remainder=False)
    neg_dataset = neg_dataset.repeat()
    neg_iter = neg_dataset.make_one_shot_iterator()
    neg_batch = neg_iter.get_next()
    batch = tf.concat([pos_batch, neg_batch], axis=0)
    neg_label = tf.zeros(shape=(tf.shape(neg_batch)[0]))
    label = tf.concat([pos_label, neg_label], axis=0)
    return batch, label


def get_test_batch(edge_set_with_label):
    dataset = tf.data.Dataset.from_tensor_slices(edge_set_with_label).batch(training_config["batch_size"],
                                                                            drop_remainder=False)
    data_iter = dataset.make_initializable_iterator()
    batch = data_iter.get_next()
    init = data_iter.initializer
    return batch, init
