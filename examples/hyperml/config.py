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
from curvlearn.manifolds import Stereographic
from curvlearn.optimizers import RAdam

training_config = {
    "manifold"             : Stereographic(),  # The manifold to be used in the model
    "init_curvature"       : 0.0,  # Default value of the curvature of the manifold
    "trainable_curvature"  : True,  # Whether the curvatures are trainable

    "data_path"            : "datasets/kindle_store/data.pkl",  # File address of the Kindle-Store dataset
    "training_steps"       : 100000,  # Steps for training
    "log_steps"            : 100,  # Log intervals during training
    "eval_steps"           : 1000,  # Evaluation intervals during training

    "batch_size"           : 512,  # Batch size used in training
    "learning_rate"        : 0.001,  # Learning rate of training
    "optimizer"            : RAdam,  # The optimizer to be used in training

    "dim"                  : 64,  # Dimension of the embedding tables
    "margin"               : 1.0,
    "candidate_num"        : 100,  # Sampled negative candidates for evaluation
    "K"                    : 10,  # Metric of HR@K

    "gamma"                : 0.2,  # Multi-task learning rate between the two types of loss
    "epsilon"              : 1e-9,  # Small value for avoiding dividing zeros in distortion optimization (loss)

    "embedding_initializer": tf.initializers.glorot_uniform(),  # Initializer for the embedding tables
    "shuffle_buffer"       : 10000,  # Buffer size for TF's shuffling over the dataset
}
