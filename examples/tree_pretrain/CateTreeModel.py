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
import tf_euler

from .utils.loss import Loss
from .utils.summary import summary_tensor
from .utils.clip_gradient import clip_gradient


class HyperCateTreeModel(object):
    def __init__(self, src_node, pos_node, global_step,
                 neg_sample_num, nb_sample_num, embedding_dimension, embedding_space, mid_dim, out_dim,
                 embedding_init_type, embedding_stddev, dense_init_type, dense_stddev, bias_init_val,
                 manifold, decode, l2_decay,
                 l2_enable, soft_c_enable, clip_gradient_enable,
                 c, loss_type, learning_rate, **kwargs):
        self.src_node = src_node
        self.pos_node = pos_node
        self.global_step = global_step

        self.neg_sample_num = neg_sample_num
        self.nb_sample_num = nb_sample_num
        self.embedding_dimension = embedding_dimension
        self.embedding_space = embedding_space
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.embedding_init_type = embedding_init_type
        self.embedding_stddev = embedding_stddev
        self.dense_init_type = dense_init_type
        self.dense_stddev = dense_stddev
        self.bias_init_val = bias_init_val

        self.manifold = manifold
        self.decode = decode
        self.l2_decay = l2_decay

        self.l2_enable = l2_enable
        self.soft_c_enable = soft_c_enable
        self.clip_gradient_enable = clip_gradient_enable

        self.init_c = tf.constant(c, dtype=self.manifold.dtype)
        self.loss_func = Loss(loss_type)
        self.opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)

        src_feature, pos_feature, neg_feature = self.cate_tree_feature_without_neighbor(self.src_node, self.pos_node)

        with tf.variable_scope('embedding_table', reuse=tf.AUTO_REUSE) as scope:
            cate_feature_names = ['node_id', 'level']
            embedding_matrix_map = {}
            for name in cate_feature_names:
                embedding_matrix_map[name] = tf.get_variable(name + '_embedding_table',
                                                             shape=(self.embedding_space, self.embedding_dimension),
                                                             dtype=self.manifold.dtype,
                                                             initializer=self.get_initializer(
                                                                 init_val=self.embedding_init_type,
                                                                 stddev=self.embedding_stddev)
                                                             )
                embedding_matrix_map[name] = self.manifold.variable(embedding_matrix_map[name], c=self.init_c)

        with tf.variable_scope('feature_embedding_layer', reuse=tf.AUTO_REUSE) as scope:
            src_embedding = self.sparse_feature_embedding(embedding_matrix_map, src_feature, cate_feature_names)
            src_embedding = self.manifold.concat(src_embedding, axis=1, c=self.init_c)

            pos_embedding = self.sparse_feature_embedding(embedding_matrix_map, pos_feature, cate_feature_names)
            pos_embedding = self.manifold.concat(pos_embedding, axis=1, c=self.init_c)

            neg_embedding_all = self.sparse_feature_embedding(embedding_matrix_map, neg_feature, cate_feature_names)
            neg_embedding_all = self.manifold.concat(neg_embedding_all, axis=1, c=self.init_c)

        if self.soft_c_enable is True:
            with tf.variable_scope('manifold_c', reuse=tf.AUTO_REUSE) as scope:
                c_mid = tf.get_variable('c_dnn', dtype=self.manifold.dtype,
                                        initializer=tf.constant(-1.0, dtype=self.manifold.dtype))
                c_out = tf.get_variable('c_dnn', dtype=self.manifold.dtype,
                                        initializer=tf.constant(-1.0, dtype=self.manifold.dtype))
        else:
            c_mid = tf.constant(-1.0, dtype=self.manifold.dtype)
            c_out = tf.constant(-1.0, dtype=self.manifold.dtype)

        clip = lambda x: tf.clip_by_value(x, clip_value_min=-1e5, clip_value_max=-1e-5)
        c_mid, c_out = clip(c_mid), clip(c_out)

        with tf.variable_scope('output_layer', reuse=tf.AUTO_REUSE) as scope:
            src_output = self.hyper_output_layer(self.init_c, c_mid, c_out, src_embedding, self.embedding_dimension * 2,
                                                 self.mid_dim, self.out_dim, scope_name='src_output')
            pos_output = self.hyper_output_layer(self.init_c, c_mid, c_out, pos_embedding, self.embedding_dimension * 2,
                                                 self.mid_dim, self.out_dim, scope_name='dst_output')
            neg_output_all = self.hyper_output_layer(self.init_c, c_mid, c_out, neg_embedding_all,
                                                     self.embedding_dimension * 2, self.mid_dim, self.out_dim,
                                                     scope_name='dst_output')

        origin = self.manifold.proj(tf.zeros([self.out_dim], dtype=self.manifold.dtype), c=c_out)
        l2_penalty = lambda x: self.manifold.distance(x, origin, c=c_out)
        penalty = []
        distance = []

        with tf.variable_scope('loss_metric_layer') as scope:
            if self.decode == 'distance':
                decode_func = lambda x, y: tf.sigmoid(5.0 - 5.0 * self.manifold.distance(x, y, c=c_out))
            else:
                decode_func = self.cosine_fun

            pos_sim = decode_func(src_output, pos_output)

            penalty.append(l2_penalty(src_output))
            penalty.append(l2_penalty(pos_output))
            distance.append(self.manifold.distance(src_output, pos_output, c=c_out))

            att_sim = [pos_sim]
            node_neg_id_ays_re = tf.reshape(neg_output_all, [-1, self.neg_sample_num * self.out_dim])
            node_neg_id_ays_list = tf.split(node_neg_id_ays_re, num_or_size_splits=self.neg_sample_num, axis=1)
            for neg in node_neg_id_ays_list:
                neg_sim = decode_func(src_output, neg)
                att_sim.append(neg_sim)

                penalty.append(l2_penalty(neg))
                distance.append(self.manifold.distance(src_output, neg, c=c_out))

        sim = tf.concat(att_sim, 1)
        tf.summary.scalar('c_final', c_out)

        l2_penalty = tf.concat(penalty, 1)
        l2_loss = tf.reduce_mean(tf.reduce_sum(l2_penalty, axis=-1))

        distance = tf.concat(distance, 1)
        pos_distance = tf.slice(distance, [0, 0], [-1, 1])
        neg_distance = tf.slice(distance, [0, 1], [-1, -1])

        summary_tensor('positive_distance', pos_distance)
        summary_tensor('negative_distance', neg_distance)
        summary_tensor('all_distance', distance)
        tf.summary.scalar('l2_penalty', l2_loss)

        self.loss = self.loss_func(sim)
        if self.l2_enable:
            self.loss += l2_decay * l2_loss

        gradients, variables = zip(*self.opt.compute_gradients(self.loss))
        if self.clip_gradient_enable:
            gradients = clip_gradient(gradients)
        self.train_op = self.opt.apply_gradients(zip(gradients, variables), global_step=self.global_step)

    def get_model_result(self):
        return self.train_op, self.loss

    def cosine_fun(self, ays_src, ays_dst):
        src_norm = tf.sqrt(tf.reduce_sum(tf.square(ays_src), 1, True))
        dst_norm = tf.sqrt(tf.reduce_sum(tf.square(ays_dst), 1, True))

        prod = tf.reduce_sum(tf.multiply(ays_src, ays_dst), 1, True)
        norm_prod = tf.multiply(src_norm, dst_norm)

        cosine = tf.truediv(prod, norm_prod)
        return cosine

    def get_initializer(self, init_val=1, stddev=0.1):
        dtype = self.manifold.dtype
        if init_val == 1:
            return tf.truncated_normal_initializer(dtype=dtype, stddev=stddev)
        elif init_val == 2:
            return tf.uniform_unit_scaling_initializer(factor=stddev, seed=10, dtype=dtype)
        elif init_val == 3:
            return tf.glorot_normal_initializer(dtype=dtype)
        else:
            return None

    def global_sample_cate_tree(self, src, pos):
        batch_size = tf.shape(src)[0]
        negs = tf_euler.sample_node(batch_size * self.neg_sample_num, node_type='-1')

        src_nodes = tf.reshape(src, [-1])
        pos_nodes = tf.reshape(pos, [-1])
        neg_nodes = tf.reshape(negs, [-1])

        return src_nodes, pos_nodes, neg_nodes

    def local_sample_cate_tree(self, src, pos):
        negs = tf_euler.sample_node_with_src(pos, self.neg_sample_num)

        src_nodes = tf.reshape(src, [-1])
        pos_nodes = tf.reshape(pos, [-1])
        neg_nodes = tf.reshape(negs, [-1])

        return src_nodes, pos_nodes, neg_nodes

    def node_feature_with_neighbor(self, node, etypes, nb_cnt, n_feature_names, cn_feature_names):
        node_c = tf.reshape(node, [-1])
        node_filled = tf_euler.get_sparse_feature(node_c, n_feature_names)

        n_nodes, _, _ = tf_euler.sample_neighbor(node, edge_types=etypes, count=nb_cnt)
        n_nodes = tf.reshape(n_nodes, [-1])
        n_nodes_filled = tf_euler.get_sparse_feature(n_nodes, cn_feature_names)

        return node_filled, n_nodes_filled

    def hyper_convolution_with_neighbor(self, node, c_node_nei, num=5, dim=8):
        c_node_nei_s = tf.reshape(c_node_nei, [-1, num, dim])
        c_nei = self.manifold.mean(c_node_nei_s, axis=1, base=None, c=self.init_c)

        features = self.manifold.concat([node, c_nei], axis=1, c=self.init_c)

        return features

    def cate_tree_feature_with_neighbor(self, source, pos_node):
        src, pos, neg = self.global_sample_cate_tree(source, pos_node)

        full_features = ['node_id', 'level']
        c_part_features = ['node_id']

        src_f, src_c_nb_f = self.node_feature_with_neighbor(node=src,
                                                            etypes=['1'],
                                                            nb_cnt=self.nb_sample_num,
                                                            n_feature_names=full_features,
                                                            cn_feature_names=c_part_features
                                                            )
        pos_f, pos_c_nb_f = self.node_feature_with_neighbor(node=pos,
                                                            etypes=['1'],
                                                            nb_cnt=self.nb_sample_num,
                                                            n_feature_names=full_features,
                                                            cn_feature_names=c_part_features
                                                            )
        neg_f, neg_c_nb_f = self.node_feature_with_neighbor(node=neg,
                                                            etypes=['1'],
                                                            nb_cnt=self.nb_sample_num,
                                                            n_feature_names=full_features,
                                                            cn_feature_names=c_part_features
                                                            )

        return src_f, src_c_nb_f, pos_f, pos_c_nb_f, neg_f, neg_c_nb_f

    def node_feature_without_neighbor(self, node, n_feature_names):
        node_c = tf.reshape(node, [-1])
        node_filled = tf_euler.get_sparse_feature(node_c, n_feature_names)

        return node_filled

    def cate_tree_feature_without_neighbor(self, source, pos_node):
        src, pos, neg = self.global_sample_cate_tree(source, pos_node)

        full_features = ['node_id', 'level']

        src_f = self.node_feature_without_neighbor(node=src, n_feature_names=full_features)
        pos_f = self.node_feature_without_neighbor(node=pos, n_feature_names=full_features)
        neg_f = self.node_feature_without_neighbor(node=neg, n_feature_names=full_features)

        return src_f, pos_f, neg_f

    def sparse_feature_embedding(self, embedding_matrix_map, sparse_inputs, names, no_biases=True):
        l = []
        for i in range(len(sparse_inputs)):
            with tf.variable_scope('sparse_feature_embedding_' + names[i]):
                emb = tf.nn.embedding_lookup_sparse(embedding_matrix_map[names[i]], sparse_inputs[i], None,
                                                    combiner='sum')

                emb_l2 = tf.nn.l2_loss(emb)
                tf.losses.add_loss(emb_l2, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

                if not no_biases:
                    biases = tf.get_variable('biases',
                                             initializer=tf.constant(self.bias_init_val, dtype=self.manifold.dtype,
                                                                     shape=[self.embedding_dimension])
                                             )
                    emb = self.manifold.add_bias(emb, biases, c=self.init_c)
            l.append(emb)
        return l

    def hyper_linear_layer(self, train_inputs, in_dim, out_dim, c_in, c_out, activation, scope_name, no_biases=False):
        with tf.variable_scope(scope_name):
            weights = tf.get_variable('weights',
                                      [in_dim, out_dim],
                                      dtype=self.manifold.dtype,
                                      initializer=self.get_initializer(init_val=self.dense_init_type,
                                                                       stddev=self.dense_stddev),
                                      regularizer=tf.nn.l2_loss
                                      )
            train = self.manifold.matmul(train_inputs, weights, c=c_in)

            if not no_biases:
                biases = tf.get_variable('biases',
                                         initializer=tf.constant(self.bias_init_val, dtype=self.manifold.dtype,
                                                                 shape=[out_dim])
                                         )
                train = self.manifold.add_bias(train, biases, c=c_in)

            if activation is not None:
                train = self.manifold.activation(train, act=activation, c_in=c_in, c_out=c_out)

            return train

    def hyper_output_layer(self, c_in, c_mid, c_out, id_embedding, input_dim, mid_dim, out_dim, scope_name):
        with tf.variable_scope(scope_name):
            out1 = self.hyper_linear_layer(train_inputs=id_embedding,
                                           in_dim=input_dim,
                                           out_dim=mid_dim,
                                           c_in=c_in,
                                           c_out=c_mid,
                                           activation=tf.nn.elu,
                                           scope_name='output_layer1',
                                           no_biases=False
                                           )

            out2 = self.hyper_linear_layer(train_inputs=out1,
                                           in_dim=mid_dim,
                                           out_dim=out_dim,
                                           c_in=c_mid,
                                           c_out=c_out,
                                           activation=tf.nn.elu,
                                           scope_name='output_layer2',
                                           no_biases=False
                                           )
            return out2
