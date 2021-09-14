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

"""Summary Util"""

import tensorflow as tf


def summary_tensor(name, sim):
    """Summary tensor.

    Args:
        name (str): The prefix name.
        sim (tensor): The tensor to statistic.
    """
    tf.summary.histogram(name + '_hist', sim)
    tf.summary.scalar(name + '_min', tf.reduce_min(sim))
    tf.summary.scalar(name + '_max', tf.reduce_max(sim))
    tf.summary.scalar(name + '_mean', tf.reduce_mean(sim))


def summary_curvature(name, sim):
    """Summary curvature.

    Args:
        name (str): The prefix name.
        sim (tensor): The tensor to statistic.
    """
    tf.summary.scalar(name + '_min', tf.reduce_min(sim))
    tf.summary.scalar(name + '_max', tf.reduce_max(sim))
    tf.summary.scalar(name + '_mean', tf.reduce_mean(sim))
