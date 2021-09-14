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

"""Manifold class with a series of neural network operations."""

import six
from curvlearn.manifolds.geometry import tf, Geometry
from abc import ABCMeta, abstractmethod


@six.add_metaclass(ABCMeta)
class Manifold(Geometry):
    """Manifold class with a series of operations for building hyperbolic neural networks
    
    Attributes:
        use_riemannian_opt (bool): Whether use riemannian optimizer or not.
    """

    def __init__(self, **kwargs):
        """Initialize a manifold.
        """
        super(Manifold, self).__init__(**kwargs)

    def variable(self, t, c):
        """Defines a riemannian variable from manifold or tangent space at origin according to its name.

        Args:
            t (tensor): A variable, can be in Euclidean space or a specific manifold.
            c (float): The manifold curvature.
        
        Returns:
            tensor: Variable on specific manifold. Has the same type with t.
        """
        if "RiemannianParameter" not in t.name:
            t = self.proj_tan0(t, c)
            t = self.expmap0(t, c)
        t = self.proj(t, c)
        return t

    def to_manifold(self, t, c, base=None):
        """Converts a tensor t in the tangent space of base point to the manifold.

        Args:
            t (tensor): A tensor, should lie in Euclidean space.
            c (float): The manifold curvature.
            base (tensor, optional): A base point on the manifold.
        
        Returns:
            tensor: A tensor in the manifold. Has the same type as t.
        """
        if base is None:
            t = self.proj_tan0(t, c)
            manifold_t = self.expmap0(t, c)
        else:
            t = self.proj_tan(t, base, c)
            manifold_t = self.expmap(t, base, c)

        return self.proj(manifold_t, c)

    def to_tangent(self, t, c, base=None):
        """Converts a tensor t in the manifold to the tangent space of base point.

        Args:
            t (tensor): A tensor, should locate in a specific manifold
            c (float): The manifold curvature.
            base (tensor, optional): A base point on the manifold.
        
        Returns:
            tensor: A tensor in Euclidean space. Has the same type as t.
        """
        if base is None:
            return self.logmap0(t, c)
        else:
            return self.logmap(t, base, c)

    def weight_sum(self, tensor_list, a, c):
        """Computes the sum of tensor list t with weight list a

        Args:
            tensor_list (list): A list of tensors. The shape of each tensor is [batch, ..., dim]
            a (list): A list of tensors as the weights. The shape of each tensor is [batch, ..., 1]
            c (float): The manifold curvature.
        
        Returns:
            tensor: The weighted sum result. The shape is [batch, ..., dim].
        """
        x, a = tf.stack(tensor_list, axis=0), tf.stack(a, axis=0)
        tangent_t = self.to_tangent(x, c=c, base=None)
        sum_t = tf.reduce_sum(a * tangent_t, axis=0)
        manifold_t = self.to_manifold(sum_t, c=c, base=None)

        return manifold_t

    def mean(self, t, c, axis=-1):
        """Computes the average of elements across dimensions of a tensor t.

        Args:
            t (tensor): The tensor to average. Should have numeric type.
            c (float): The manifold curvature.
            axis (int, optional): The dimensions to average.
        
        Returns:
            tensor: A tensor resulting from averaging of the input tensor. Lies in the same manifold as t.
        """
        t = tf.unstack(t, axis=axis)
        w = [tf.ones_like(tf.reduce_sum(_t, keepdims=True, axis=-1))
             for _t in t]
        w = [_w / float(len(t)) for _w in w]

        return self.weight_sum(t, w, c)

    def sum(self, t, c, axis=-1):
        """Computes the sum of elements across dimensions of a tensor t.

        Args:
            t (tensor): The tensor to sum. Should have numeric type.
            c (float): The manifold curvature.
            axis (int, optional): The dimensions to sum.
        
        Returns:
            tensor: A tensor resulting from summation of the input tensor. Lies in the same manifold as t.
        """
        t = tf.unstack(t, axis=axis)
        w = [tf.ones_like(tf.reduce_sum(_t, keepdims=True, axis=-1))
             for _t in t]

        return self.weight_sum(t, w, c)

    def concat(self, tensor_list, c, axis=-1):
        """Concatenates tensors along one dimension.

        Args:
            tensor_list (list): A list of Tensor objects.
            c (float): The manifold curvature.
            axis (int, optional): Dimension along which to concatenate.
        
        Returns:
            tensor: A tensor resulting from concatenation of the input tensors. Lies in the same manifold as t.
        """
        tensor_list = list(map(
            lambda x: self.to_tangent(x, c=c, base=None),
            tensor_list
        ))
        concat_tensor = tf.concat(tensor_list, axis=axis)
        concat_tensor = self.to_manifold(concat_tensor, c=c, base=None)

        return concat_tensor

    def add(self, x, y, c):
        """Adds tensor x and tensor y.

        Args:
            x (tensor): A tensor.
            y (tensor): A tensor. Must have the same type as x.
            c (float): The manifold curvature.
        
        Returns:
            tensor: A tensor. Lies in the same manifold as t.
        """
        if hasattr(self, "_mobius_add"):
            return self._mobius_add(x, y, c)
        return self.sum(tf.stack([x, y], axis=0), c=c, axis=0)

    def matmul(self, t, m, c):
        """Multiplies tensor t by euclidean matrix m.

        Args:
            t (tensor): A tensor.
            m (tensor): The parameter matrix, should lie in Euclidean.
            c (float): The manifold curvature.
        
        Returns:
            tensor: A tensor. Lies in the same manifold as t.
        """
        x = self.to_tangent(t, c=c, base=None)
        mx = tf.matmul(x, m)
        mx = self.to_manifold(mx, c=c, base=None)
        return mx

    def add_bias(self, t, b, c):
        """Adds a euclidean bias vector b to tensor t.

        Args:
            t (tensor): A tensor.
            b (tensor): The parameter bias, should lie in Euclidean.
            c (float): The manifold curvature.
        
        Returns:
            tensor: A tensor. Lies in the same manifold as t.
        """
        b = self.proj_tan0(b, c)
        trans_b = self.ptransp0(b, t, c)
        t = self.to_manifold(trans_b, c=c, base=t)

        return t

    def activation(self, t, c_in, c_out, act):
        """Computes the non-linear activation value for the input tensor t.

        Args:
            t (tensor): A tensor.
            c_in (float): The manifold curvature for the input layer.
            c_out (float): The manifold curvature for the output layer.
            act (function): The non-linear activation function.
        
        Returns:
            tensor: The result tensor after non linear activation.
        """
        t = self.to_tangent(t, c=c_in, base=None)
        act_out = act(t)
        out = self.to_manifold(act_out, c=c_out, base=None)

        return out

    def linear(self, t, in_dim, out_dim, c_in, c_out, act, scope="linear"):
        """Computes the linear transformation and activation for the input tensor t.

        Args:
            t (tensor): A tensor.
            in_dim (int): The dimension of the input tensor.
            out_dim (int): The dimension of the output tensor.
            c_in (float): The manifold curvature for the input tensor.
            c_out (float): The manifold curvature for the output tensor.
            act (function): The non-linear activation function.
            scope (str, optional): the scope name for the variable.
        
        Returns:
            tensor: The result tensor after linear transformation and activation.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            w = tf.get_variable(
                "weights",
                shape=[in_dim, out_dim],
                dtype=self.dtype,
                initializer=tf.glorot_normal_initializer(dtype=self.dtype)
            )
            b = tf.get_variable(
                "bias",
                shape=[out_dim],
                dtype=self.dtype,
                initializer=tf.random_uniform_initializer(
                    minval=-1e-3, maxval=1e-3)
            )
            mx = self.matmul(t, w, c=c_in)
            mx = self.add_bias(mx, b, c=c_in)

            if act is not None:
                mx = self.activation(mx, act=act, c_in=c_in, c_out=c_out)

        return mx

    def distance(self, src, tar, c):
        """Computes the squared geodesic/distance between src and tar.

        Args:
            src (tensor): The source point.
            tar (tensor): The target point. Lies in the same manifold as src.
            c (float): The manifold curvature.
        
        Returns:
            tensor: The distance between src and tar.
        """
        src, tar = self.proj(src, c), self.proj(tar, c)
        dist = self.sqdist(src, tar, c)
        dist = tf.expand_dims(dist, -1)
        return dist
