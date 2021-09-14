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
from curvlearn.manifolds import PoincareBall

training_config = {
    "manifold"           : PoincareBall(),  # The manifold to be used in the model
    "init_curvature"     : -1.0,  # Default value of the curvature of the manifold
    "trainable_curvature": True,  # Whether the curvatures are trainable

    "training_steps"     : 100000,  # Steps for training
    "log_steps"          : 100,  # Log intervals during training
    "eval_steps"         : 1000,  # Evaluation intervals during training

    "batch_size"         : 14905,  # Batch size used in training
    "learning_rate"      : 0.01,  # Learning rate of training
    "dropout_rate"       : 0.0,  # The ratio to discard value
    "optimizer"          : tf.train.AdamOptimizer,  # The optimizer to be used in training

    "hidden_layer_dims"  : [16, 16],  # The dimension of the hidden layers
    "neg_samples"        : 1,  # Number of negative samples towards per positive samples

    "initializer"        : tf.initializers.glorot_uniform(),  # The initializer of the embedding tables
    "normalize_adj"      : True,  # Whether to normalize the adjacency matrix
    "normalize_feat"     : True,  # Whether to normalize the feature

    "shuffle_buffer"     : 5070224,  # Buffer size for TF's shuffling over the dataset

}
