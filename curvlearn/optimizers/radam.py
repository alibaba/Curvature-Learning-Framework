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

"""Riemannian Adam optimizer"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, state_ops, control_flow_ops, gen_sparse_ops

from curvlearn.manifolds import euclidean


class RAdam(tf.train.AdamOptimizer):
    """Riemannian Adam optimizer.
    """

    def __init__(self,
                 manifold,
                 c,
                 learning_rate=1e-3,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 use_locking=False,
                 name="RAdam"):
        """Riemannian Adam optimizer initialization.

        Args:
            manifold (class): The manifold.
            c (float): The manifold curvature.
            learning_rate (float, optional): The learning rate to use.
            beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            epsilon (float, optional): A small constant for numerical stability.
            use_locking (bool, optional): If True use locks for update operations.
            name (str, optional): The optimizer name.
        """
        super(RAdam, self).__init__(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
                                    use_locking=use_locking, name=name)

        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self.default_manifold = euclidean.Euclidean()

        self.manifold = manifold
        self.default_c = c

    def _get_beta_accumulators(self):
        with ops.init_scope():
            graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph))

    def _prepare(self):
        learning_rate = self._call_if_callable(self._learning_rate)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)

        self._lr_t = ops.convert_to_tensor(
            learning_rate, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

    def _create_slots(self, var_list):
        """Create the beta1 and beta2 accumulators on the same device as the first
        variable. Sort the var_list to make sure this device is consistent across
        workers (these need to go on the same PS, otherwise some updates are
        silently ignored).
        """
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

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
        if var.manifold.name == "Euclidean":
            return super(RAdam, self)._apply_dense(grad, var)

        beta1_power, beta2_power = self._get_beta_accumulators()

        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        learning_rate = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1 = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2 = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (learning_rate * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        rgrad = var.manifold.egrad2rgrad(grad, var, c=self.default_c)
        rgrad_sq = var.manifold.inner(
            rgrad, rgrad, var, c=self.default_c, keep_shape=True)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        scaled_m = m * beta1 + rgrad * (1 - beta1)
        m_t = state_ops.assign(m, scaled_m, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        scaled_v = v * beta2 + rgrad_sq * (1 - beta2)
        v_t = state_ops.assign(v, scaled_v, use_locking=self._use_locking)

        # x_t = x - lr * m_t / sqrt(v_t)
        with tf.control_dependencies([m_t, v_t]):
            new_value = var.manifold.retraction(-lr * m_t / (math_ops.sqrt(v_t) + epsilon),
                                                var, c=self.default_c)
            var_update = state_ops.assign(
                ref=var, value=new_value, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        if var.manifold.name == "Euclidean":
            return super(RAdam, self)._apply_sparse(grad, var)

        raise NotImplementedError

    def _resource_apply_dense(self, grad, var):
        if var.manifold.name == "Euclidean":
            return super(RAdam, self)._resource_apply_dense(grad, var)

        raise NotImplementedError

    def _resource_apply_sparse(self, grad, var, indices):
        if var.manifold.name == "Euclidean":
            return super(RAdam, self)._resource_apply_sparse(grad, var, indices)

        raise NotImplementedError

    def _finish(self, update_ops, name_scope):
        """Update the power accumulators.
        """
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(
            *update_ops + [update_beta1, update_beta2], name=name_scope)
