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
from curvlearn.manifolds import euclidean

from config import training_config


def getCurvature(name, manifold, init_val=training_config["init_curvature"]):
    with tf.variable_scope("curvature", reuse=tf.AUTO_REUSE) as scope:
        return tf.get_variable(
            name=name,
            dtype=manifold.dtype,
            initializer=tf.constant(init_val, dtype=manifold.dtype),
            trainable=training_config["trainable_curvature"]
        )


class HyperMLModel(object):
    def __init__(self, user_count, item_count, bpr_pair, is_training=True):
        self._user_count = user_count
        self._item_count = item_count
        self._dim = training_config["dim"]
        self._manifold = training_config["manifold"]
        self._eclidean_manifold = euclidean.Euclidean()
        self._input_ids = bpr_pair
        self._is_training = is_training
        self._c = getCurvature("hyperml_curvature", self._manifold)
        with tf.variable_scope("hyperml", reuse=tf.AUTO_REUSE):
            self.embedding_table_user = tf.get_variable(name="RiemannianParameter/user_embedding", shape=(self._user_count, self._dim),
                                                        dtype=tf.float32,
                                                        initializer=training_config["embedding_initializer"])
            self.embedding_table_user = self._manifold.variable(self.embedding_table_user, self._c)
            self.embedding_table_item = tf.get_variable(name="RiemannianParameter/item_embedding", shape=(self._item_count, self._dim),
                                                        dtype=tf.float32,
                                                        initializer=training_config["embedding_initializer"])
            self.embedding_table_item = self._manifold.variable(self.embedding_table_item, self._c)
        if self._is_training:
            self.uid, self.pos_iid, self.neg_iid = tf.split(self._input_ids, 3, axis=1)
            self.user_vectors = tf.nn.embedding_lookup(self.embedding_table_user, self.uid)
            self.pos_item_vectors = tf.nn.embedding_lookup(self.embedding_table_item, self.pos_iid)
            self.neg_item_vectors = tf.nn.embedding_lookup(self.embedding_table_item, self.neg_iid)
            self._dist_pos = self._manifold.distance(self.user_vectors, self.pos_item_vectors, self._c)
            self._dist_neg = self._manifold.distance(self.user_vectors, self.neg_item_vectors, self._c)
            self.loss_p = tf.reduce_mean(tf.nn.relu(
                tf.reduce_sum(self._dist_pos, axis=-1) - tf.reduce_sum(self._dist_neg, axis=-1) +
                training_config["margin"]))
            d_Dij = self._dist_pos
            d_Eij = self._eclidean_manifold.distance(self._manifold.to_tangent(self.user_vectors, self._c),
                                                     self._manifold.to_tangent(self.pos_item_vectors, self._c), self._c)
            d_Dik = self._dist_neg
            d_Eik = self._eclidean_manifold.distance(self._manifold.to_tangent(self.user_vectors, self._c),
                                                     self._manifold.to_tangent(self.neg_item_vectors, self._c), self._c)
            self.loss_d1 = tf.reduce_mean(tf.nn.relu(tf.abs(d_Dij**0.5 - d_Eij**0.5) / (d_Eij**0.5 + training_config["epsilon"])))
            self.loss_d2 = tf.reduce_mean(tf.nn.relu(tf.abs(d_Dik**0.5 - d_Eik**0.5) / (d_Eik**0.5 + training_config["epsilon"])))
            self.loss = self.loss_p + training_config["gamma"] * (self.loss_d1 + self.loss_d2)
            self.optimizer = training_config["optimizer"](learning_rate=training_config["learning_rate"],
                                                          manifold=self._manifold, c=self._c)
            self.train_op = self.optimizer.minimize(self.loss)
        else:
            self.uid, self.can_iid = tf.split(self._input_ids, [1, training_config["candidate_num"] + 1], axis=1)
            self.user_vectors = tf.tile(tf.nn.embedding_lookup(self.embedding_table_user, self.uid),
                                        [1, training_config["candidate_num"] + 1, 1])
            self.can_item_vectors = tf.nn.embedding_lookup(self.embedding_table_item, self.can_iid)
            self.predict_scores = tf.squeeze(
                self._manifold.distance(self.user_vectors, self.can_item_vectors, self._c))
            self.rank_items = tf.argsort(self.predict_scores)[:, :training_config["K"]]
            self.correct = tf.get_variable(name="correct", shape=(), dtype=tf.int32,
                                           initializer=tf.zeros_initializer(dtype=tf.int32))
            self.total = tf.get_variable(name="total", shape=(), dtype=tf.int32,
                                         initializer=tf.zeros_initializer(dtype=tf.int32))
            self.metric_initializers = [self.correct.initializer, self.total.initializer]
            self.updata_correct = tf.assign_add(self.correct,
                                                tf.reduce_sum(tf.cast(tf.equal(self.rank_items, 0), tf.int32)))
            self.update_total = tf.assign_add(self.total, tf.shape(self.uid)[0])
            with tf.control_dependencies([self.updata_correct, self.update_total]):
                self.hr_at_k = tf.cast(self.correct, tf.float32) / tf.cast(self.total, tf.float32)
