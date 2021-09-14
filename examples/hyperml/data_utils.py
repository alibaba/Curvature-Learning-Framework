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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('.')

import numpy as np
import pickle as pkl
import random as rd
import tensorflow as tf

from config import training_config


user_count, item_count = 0, 0
pos_set = None


def load_data(data_path):
    fin = open(data_path, 'rb')
    data = pkl.load(fin)
    fin.close()
    global user_count, item_count, pos_set
    user_count = len(data) + 1
    item_count = max([max(_) for _ in data]) + 1
    train_data, test_data = [], []
    for uid, item_list in enumerate(data):
        for i in item_list[:-1]:
            train_data.append((uid + 1, i))
        test_data.append((uid + 1, item_list[-1]))
    record_count = len(train_data)
    print("[TRAINING] Loading data completed! User count: {}, item count: {}, training record count: {}".format(
        user_count, item_count, record_count))
    pos_set = set(train_data)
    train_data = np.array(train_data)
    np.random.shuffle(train_data)
    return train_data, np.array(test_data), user_count, item_count


def construct_batch(ui_pair):
    while True:
        neg_i = rd.randint(1, item_count - 1)
        if (ui_pair[0], neg_i) not in pos_set:
            break
    return np.array((list(ui_pair) + [neg_i]), dtype=np.int32)


def construct_test(ui_pair):
    candidate_set = {ui_pair[1]}
    for _ in range(training_config["candidate_num"]):
        while True:
            neg_i = rd.randint(1, item_count - 1)
            if neg_i not in candidate_set and (ui_pair[0], neg_i) not in pos_set:
                candidate_set.add(neg_i)
                break
    candidate_set.remove(ui_pair[1])
    return np.array(list(ui_pair) + list(candidate_set), dtype=np.int32)


def get_train_batch(data):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(lambda x: tf.py_func(construct_batch, [x], tf.int32))
    dataset = dataset.shuffle(buffer_size=training_config["shuffle_buffer"]).batch(training_config["batch_size"],
                                                                                   drop_remainder=False).repeat()
    data_iter = dataset.make_one_shot_iterator()
    batch = data_iter.get_next()
    return batch


def get_test_batch(data):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(lambda x: tf.py_func(construct_test, [x], tf.int32))
    dataset = dataset.batch(training_config["batch_size"])
    data_iter = dataset.make_initializable_iterator()
    init = data_iter.initializer
    batch = data_iter.get_next()
    return batch, init
