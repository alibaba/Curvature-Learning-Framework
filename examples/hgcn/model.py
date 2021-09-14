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

import tensorflow as tf
import math
from config import training_config


def getCurvature(name, manifold, init_val=training_config["init_curvature"]):
    with tf.variable_scope("curvature", reuse=tf.AUTO_REUSE) as scope:
        return tf.get_variable(
            name=name,
            dtype=manifold.dtype,
            initializer=tf.constant(init_val, dtype=manifold.dtype),
            trainable=training_config["trainable_curvature"]
        )


class HGCNModel(object):
    def __init__(self, node_count, feature, adj_train, input_ids, label, is_training=True):
        self._node_count = node_count
        self._manifold = training_config["manifold"]
        self._is_training = is_training
        self._input_ids = input_ids
        self._label = label
        self._c = getCurvature("model_curvature", self._manifold)
        self._layers_c = [getCurvature("layer_curvature_{}".format(i), self._manifold) for i in
                          range(len(training_config["hidden_layer_dims"]))]
        self.build_model(feature, adj_train)

    def build_model(self, feature, adj_train):
        with tf.variable_scope("hgcn", reuse=tf.AUTO_REUSE):
            self.gcn_layer(feature, adj_train)
            self.logit_layer()
            self.entropy_loss()
            self.metric_value()

    def gcn_layer(self, feature, adj_train):
        self._feature = tf.constant(feature)
        self._feature = self._manifold.variable(self._feature, c=self._layers_c[0])
        self._adj_train = tf.constant(adj_train)
        layer_input = self._feature
        dims = [self._feature.shape[-1]] + training_config["hidden_layer_dims"]
        c_list = self._layers_c + [self._c]
        for i in range(1, len(dims)):
            with tf.variable_scope("fc_{}".format(i - 1), reuse=tf.AUTO_REUSE):
                weight = tf.get_variable(name="kernels", dtype=tf.float32, shape=(dims[i - 1], dims[i]),
                                         initializer=training_config["initializer"]) * math.sqrt(2)
                if self._is_training and training_config["dropout_rate"] > 0:
                    weight = tf.nn.dropout(weight, rate=training_config["dropout_rate"])
                bias = tf.get_variable(name="biases", dtype=tf.float32, shape=(1, dims[i]),
                                       initializer=tf.zeros_initializer(dtype=tf.float32))
                mv = self._manifold.matmul(layer_input, weight, c=c_list[i - 1])
                layer_input = self._manifold.add_bias(mv, bias, c=c_list[i - 1])
                layer_input = self._manifold.to_tangent(layer_input, c=c_list[i - 1])
                layer_input = tf.matmul(self._adj_train, layer_input)
                layer_input = tf.nn.relu(layer_input)
                layer_input = self._manifold.to_manifold(layer_input, c=c_list[i])
        self._gcn_output = layer_input

    def logit_layer(self):
        self._decode_x = tf.nn.embedding_lookup(self._gcn_output, self._input_ids[:, 0])
        self._decode_y = tf.nn.embedding_lookup(self._gcn_output, self._input_ids[:, 1])
        self._dist = self._manifold.distance(self._decode_x, self._decode_y, self._c)
        self._dist = tf.clip_by_value(self._dist, 0, 50)  # In case the gradients being NaN
        self._logit = 1. / (tf.exp(self._dist - 2.) + 1.)

    def entropy_loss(self):
        self.loss = tf.keras.losses.binary_crossentropy(tf.reshape(self._label, [-1, 1]),
                                                        tf.reshape(self._logit, [-1, 1]))
        self.loss = tf.reduce_mean(self.loss)
        self.optimizer = training_config["optimizer"](learning_rate=training_config["learning_rate"])
        self.train_op = self.optimizer.minimize(self.loss)

    def metric_value(self):
        _, self.auc = tf.metrics.auc(self._label, self._logit, name="train_auc" if self._is_training else "test_auc")
        if self._is_training:
            self.metric_initializers = [v.initializer for v in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES) if
                                        "train_auc" in v.name]
        else:
            self.metric_initializers = [v.initializer for v in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES) if
                                        "test_auc" in v.name]
