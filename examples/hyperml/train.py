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

import tensorflow as tf

from config import training_config
from data_utils import load_data, get_train_batch, get_test_batch
from model import HyperMLModel


def evaluate(sess, test_model, test_init):
    sess.run(test_init)
    sess.run(test_model.metric_initializers)
    while True:
        try:
            metric_value = sess.run(test_model.hr_at_k)
        except tf.errors.OutOfRangeError:
            break
    return metric_value


def train():
    train_data, test_data, user_count, item_count = load_data(training_config["data_path"])
    train_batch = get_train_batch(train_data)
    test_batch, test_init = get_test_batch(test_data)
    train_model = HyperMLModel(user_count, item_count, train_batch, True)
    test_model = HyperMLModel(user_count, item_count, test_batch, False)
    global_init = tf.global_variables_initializer()
    cp = tf.ConfigProto()
    cp.gpu_options.allow_growth = True
    sess = tf.train.MonitoredTrainingSession(config=cp)
    sess.run(global_init)
    for i in range(training_config["training_steps"]):
        loss, _ = sess.run([train_model.loss, train_model.train_op])
        if i % training_config["log_steps"] == 0 or i == training_config["training_steps"] - 1:
            print("[TRAINING] step[{}]: loss[{}]".format(i, loss))
        if i and i % training_config["eval_steps"] == 0:
            eval_str = "[TRAINING] step[{}]: eval HR@{}[{}]".format(i, training_config["K"],
                                                                    evaluate(sess, test_model, test_init))
            print(eval_str)
    print("[TRAINING] Train end!")
    print("[TRAINING] Final HR@{}: {}".format(training_config["K"], evaluate(sess, test_model, test_init)))


if __name__ == "__main__":
    train()
