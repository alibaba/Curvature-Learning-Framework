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

import tensorflow as tf
import tf_euler
from CateTreeModel import HyperCateTreeModel
from curvlearn.manifolds import PoincareBall


def define_network_flags():
    tf.flags.DEFINE_string('euler_dir', 'datasets/cate_tree/cate_tree', 'Euler Graph.')
    tf.flags.DEFINE_string('sample_dir', 'datasets/cate_tree/cate_tree.csv', 'Train Sample.')

    tf.flags.DEFINE_integer('epochs', 10000, 'Epochs to train')
    tf.flags.DEFINE_integer('batch_size', 1024, 'Mini-batch size.')
    tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
    tf.flags.DEFINE_float('l2_decay', 1e-3, 'L2 Loss.')
    tf.flags.DEFINE_float('c', -1.0, 'Curvature.')
    tf.flags.DEFINE_integer('log_steps', 100, 'Log intervals during training.')

    tf.flags.DEFINE_integer('neg_sample_num', 6, 'Negative sample number.')
    tf.flags.DEFINE_integer('nb_sample_num', 6, 'Neighbor sample number.')
    tf.flags.DEFINE_integer('embedding_dimension', 64, 'Embedding dimension.')
    tf.flags.DEFINE_integer('embedding_space', 100000, 'Embedding Space.')
    tf.flags.DEFINE_integer('mid_dim', 1024, 'Hidden dimension.')
    tf.flags.DEFINE_integer('out_dim', 64, 'Output dimension.')

    tf.flags.DEFINE_integer('embedding_init_type', 1, 'Embedding Initialization.')
    tf.flags.DEFINE_float('embedding_stddev', 0.035, 'Embedding Standard Deviation.')
    tf.flags.DEFINE_integer('dense_init_type', 3, 'Dense Weight Initialization.')
    tf.flags.DEFINE_float('dense_stddev', 0.01, 'Dense Weight Standard Deviation.')
    tf.flags.DEFINE_float('bias_init_val', 0.0002, 'Bias Initialization.')

    tf.flags.DEFINE_boolean('l2_enable', True, 'Use L2 Regularization.')
    tf.flags.DEFINE_boolean('soft_c_enable', False, 'Use Soft C.')
    tf.flags.DEFINE_boolean('clip_gradient_enable', True, 'Use Clip Gradient.')

    tf.flags.DEFINE_enum('loss_type', 'triplet_loss', ['bce_loss', 'triplet_loss', 'ranking_loss', 'bpr_loss'],
                         'Loss Function.')
    tf.flags.DEFINE_enum('decode', 'distance', ['distance', 'cosine'], 'Distance Decode.')


def main(_):
    FLAGS = tf.flags.FLAGS
    tf_euler.initialize_embedded_graph(FLAGS.euler_dir)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    src_id = []
    dst_id = []
    level = []

    with open(FLAGS.sample_dir, 'r') as w:
        for line in w.read()[:-1].split('\n'):
            line_str = line.split(',')
            src_id.append(int(line_str[0]))
            dst_id.append(int(line_str[1]))
            level.append(int(line_str[2]))

    print('Sample number', len(src_id))
    dataset = tf.data.Dataset.from_tensor_slices({'src_id': tf.cast(src_id, tf.int64),
                                                  'dst_id': tf.cast(dst_id, tf.int64),
                                                  'level': tf.cast(level, tf.int64)}
                                                 )
    print('Dataset output', dataset.output_types, dataset.output_shapes)
    dataset = dataset.shuffle(FLAGS.batch_size * 10).batch(FLAGS.batch_size, drop_remainder=False).repeat(FLAGS.epochs)
    print('RepeatDataset output', dataset.output_types, dataset.output_shapes)

    iterator = tf.data.make_one_shot_iterator(dataset)
    batch = iterator.get_next()
    src, pos, level = batch['src_id'], batch['dst_id'], batch['level']
    cate_tree_model = HyperCateTreeModel(src,
                                         pos,
                                         global_step,
                                         FLAGS.neg_sample_num,
                                         FLAGS.nb_sample_num,
                                         FLAGS.embedding_dimension,
                                         FLAGS.embedding_space,
                                         FLAGS.mid_dim,
                                         FLAGS.out_dim,

                                         FLAGS.embedding_init_type,
                                         FLAGS.embedding_stddev,
                                         FLAGS.dense_init_type,
                                         FLAGS.dense_stddev,
                                         FLAGS.bias_init_val,

                                         PoincareBall(),
                                         FLAGS.decode,
                                         FLAGS.l2_decay,

                                         FLAGS.l2_enable,
                                         FLAGS.soft_c_enable,
                                         FLAGS.clip_gradient_enable,

                                         FLAGS.c,
                                         FLAGS.loss_type,
                                         FLAGS.learning_rate
                                         )

    train_op, loss = cate_tree_model.get_model_result()
    ops = [train_op, loss] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    batch_idx = 0
    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    cp = tf.ConfigProto()
    cp.gpu_options.allow_growth = True

    with tf.Session(config=cp) as sess:
        sess.run([global_init, local_init])
        while True:
            try:
                batch_idx += 1
                _, loss = sess.run(ops)
                if batch_idx % FLAGS.log_steps == 1:
                    print('No.{} batches, loss {}'.format(batch_idx, loss))


            except tf.errors.OutOfRangeError:
                print('Finish train')
                break


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_network_flags()
    tf.app.run(main)
