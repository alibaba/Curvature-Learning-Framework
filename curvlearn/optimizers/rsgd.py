# coding=utf-8

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

"""Riemannian SGD optimizer"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, state_ops, control_flow_ops

from curvlearn.manifolds import euclidean


class RSGD(tf.train.GradientDescentOptimizer):
    """Reimannian SGD optimizer
    """

    def __init__(self,
                 manifold,
                 c,
                 learning_rate=1e-3,
                 use_locking=False,
                 name="RSGD"):
        """Riemannian SGD optimizer initialization.

        Args:
            manifold (class): The manifold.
            c (float): The manifold curvature.
            learning_rate (float, optional): The learning rate to use.
            use_locking (bool, optional): If True use locks for update operations.
            name (str, optional): The optimizer name.
        """
        super(RSGD, self).__init__(learning_rate=learning_rate, use_locking=use_locking, name=name)

        self._learning_rate = learning_rate
        self.default_manifold = euclidean.Euclidean()

        self.manifold = manifold
        self.default_c = c

    def _prepare(self):
        """Converts learning rate to tensor.
        """
        learning_rate = self._call_if_callable(self._learning_rate)
        self._learning_rate_tensor = ops.convert_to_tensor(
            learning_rate, name="learning_rate")

    def _create_slots(self, var_list):
        """Creates manifold slot for var_list.

        Args:
            var_list (list): A list of variables.
        """
        for v in var_list:
            # TODO: pass manifold attr into trainable_variable list
            # v.manifold = v.manifold if hasattr(v, "manifold") else self.default_manifold
            if "RiemannianParameter" in v.name:
                v.manifold = self.manifold
            else:
                v.manifold = self.default_manifold

    def _apply_dense(self, grad, var):
        """Apply gradients to variables.

        Args:
            grad (tensor): The gradient.
            var (tensor): The variable.

        Returns:
            operation: An Operation that applies the specified gradients.
        """
        rgrad = var.manifold.egrad2rgrad(grad, var, c=self.default_c)
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)

        new_value = var.manifold.retraction(-lr * rgrad, var, c=self.default_c)
        var_update = state_ops.assign(
            ref=var, value=new_value, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update])

    def _apply_sparse(self, grad, var):
        if var.manifold.name == "Euclidean":
            return super(RSGD, self)._apply_sparse(grad, var)

        raise NotImplementedError

    def _resource_apply_dense(self, grad, var):
        if var.manifold.name == "Euclidean":
            return super(RSGD, self)._resource_apply_dense(grad, var)

        raise NotImplementedError

    def _resource_apply_sparse(self, grad, var, indices):
        if var.manifold.name == "Euclidean":
            return super(RSGD, self)._resource_apply_sparse(grad, var, indices)

        raise NotImplementedError
