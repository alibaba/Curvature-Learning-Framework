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

"""Gradient Clipping"""

import tensorflow as tf


def clip_gradient(grad_list, clip_norm=10):
    """Clip gradient.
    
    If gradient encounters NaN, set all gradients to 0. Then gather all
    gradients, restrict their total norm under clip_norm.

    Args:
        grad_list (list): The gradient list.
        clip_norm (int, optional): A maximum clipping value. Defaults 
            to 10.

    Returns:
        list: The clipped gradient list.
    """

    def _safe_where(cond, x, y):
        """Return x if the predicate cond is true else y.
        
        Args:
            cond (tensor): The tensor determining whether to return x or y.
            x (tensor): The tensor returned if cond is true.
            y (tensor): The tensor returned if cond is false.
        
        Returns:
            tensor: x if cond is true, else y.
        """
        # https://github.com/tensorflow/tensorflow/issues/20091
        return tf.cond(cond, lambda: x, lambda: y)

    def _has_nan(grad):
        """Determine whether grad encounters nan.
        
        Args:
            grad (tensor): The input gradient.
        
        Returns:
            tensor: The tensor determining whether grad encounters nan.
        """
        if grad is None:
            return False
        elif isinstance(grad, tf.IndexedSlices):
            return tf.reduce_any(tf.is_nan(grad.values))
        else:
            return tf.reduce_any(tf.is_nan(grad))

    def _clip_gradient(grad, cond):
        """Clips gradient if grad has nan values.
        
        Args:
            grad (tensor): The input gradient.
            cond (tensor): The tensor determining whether grad encounters nan.
        
        Returns:
            tensor: The clipped gradient
        """
        if grad is None:
            return grad
        elif isinstance(grad, tf.IndexedSlices):
            grad_values = _safe_where(
                cond, tf.zeros_like(grad.values), grad.values)
            grad_values = tf.clip_by_norm(grad_values, clip_norm=clip_norm)

            return tf.IndexedSlices(
                grad_values,
                grad.indices,
                grad.dense_shape
            )
        else:
            grad = _safe_where(cond, tf.zeros_like(grad), grad)
            grad = tf.clip_by_norm(grad, clip_norm=clip_norm)
            return grad

    grad_has_nan = tf.reduce_any(list(map(_has_nan, grad_list)))
    grad_list = list(map(lambda x: _clip_gradient(x, grad_has_nan), grad_list))
    return grad_list
