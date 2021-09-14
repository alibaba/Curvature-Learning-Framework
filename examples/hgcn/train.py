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

import sys
sys.path.append('.')

import tensorflow as tf

from config import training_config
from data_utils import load_data_airport, get_train_batch, get_test_batch
from model import HGCNModel


def evaluate(sess, eval_model, eval_init):
    sess.run([eval_init])
    sess.run(eval_model.metric_initializers)
    while True:
        try:
            loss, auc = sess.run([eval_model.loss, eval_model.auc])
        except tf.errors.OutOfRangeError:
            break
    return auc


def train():
    data = load_data_airport('datasets/airport', training_config["normalize_adj"], training_config["normalize_feat"])
    node_count, feat_dim = data['features'].shape
    nb_false_edges = len(data['train_edges_false'])
    nb_edges = len(data['train_edges'])
    print("[TRAINING] Node count: {}".format(node_count))
    train_batch, train_label = get_train_batch(data['train_edges'], data['train_edges_false'])
    eval_batch, eval_init = get_test_batch(data['val_edges'])

    train_model = HGCNModel(node_count, data['features'], data['adj_train_norm'], train_batch, train_label, True)
    eval_model = HGCNModel(node_count, data['features'], data['adj_train_norm'], eval_batch[:, :2],
                           tf.cast(eval_batch[:, 2], dtype=tf.float32), False)

    global_init = tf.global_variables_initializer()
    cp = tf.ConfigProto()
    cp.gpu_options.allow_growth = True
    sess = tf.train.MonitoredTrainingSession(config=cp)
    sess.run(global_init)
    for i in range(training_config["training_steps"]):
        _, train_loss = sess.run([train_model.train_op, train_model.loss])
        if i % training_config["log_steps"] == 0 or i == training_config["training_steps"] - 1:
            print("[TRAINING] step[{}]: loss[{}]".format(i, train_loss))
        if i and i % training_config["eval_steps"] == 0:
            print("[TRAINING] step[{}]: eval AUC[{}]".format(i, evaluate(sess, eval_model, eval_init)))
    print("[TRAINING] Train finished!")
    print("[TRAINING] Final AUC: {}".format(evaluate(sess, eval_model, eval_init)))


if __name__ == "__main__":
    train()
