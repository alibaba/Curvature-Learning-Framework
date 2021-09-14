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

"""Abstract class to define basic operations on a manifold."""

import six
import tensorflow as tf
from abc import ABCMeta, abstractmethod


@six.add_metaclass(ABCMeta)
class Geometry(object):
    """Abstract class to define basic operations on a manifold.
    
    Attributes:
        clip (function): Clips tensor values to a specified min and max.
        dtype: The type of the variables.
        eps (float): A small constant value.
        max_norm (float): The maximum value for number clipping.
        min_norm (float): The minimum value for number clipping.
    """

    def __init__(self, **kwargs):
        """Initialize a manifold.
        """
        super(Geometry, self).__init__()

        self.min_norm = 1e-10
        self.max_norm = 1e10
        self.eps = 1e-5

        self.dtype = kwargs["dtype"] if "dtype" in kwargs else tf.float32
        self.clip = lambda x: tf.clip_by_value(x, clip_value_min=self.min_norm, clip_value_max=self.max_norm)

    @abstractmethod
    def proj(self, x, c):
        """A projection function that prevents x from leaving the manifold.

        Args:
            x (tensor): A point should be on the manifold, but it may not meet the manifold constraints.
            c (float): The manifold curvature.

        Returns:
            tensor: A projected point, meeting the manifold constraints.
        """

    @abstractmethod
    def proj_tan(self, v, x, c):
        """A projection function that prevents v from leaving the tangent space of point x.

        Args:
            v (tensor): A point should be on the tangent space, but it may not meet the manifold constraints.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: A projected point, meeting the tangent space constraints.
        """

    @abstractmethod
    def proj_tan0(self, v, c):
        """A projection function that prevents v from leaving the tangent space of origin point.

        Args:
            v (tensor): A point should be on the tangent space, but it may not meet the manifold constraints.
            c (float): The manifold curvature.

        Returns:
            tensor: A projected point, meeting the tangent space constraints.
        """

    @abstractmethod
    def expmap(self, v, x, c):
        """Map a point v in the tangent space of point x to the manifold.

        Args:
            v (tensor): A point in the tangent space of point x.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of mapping tangent point v to the manifold.
        """

    @abstractmethod
    def expmap0(self, v, c):
        """Map a point v in the tangent space of origin point to the manifold.

        Args:
            v (tensor): A point in the tangent space of origin point.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of mapping tangent point v to the manifold.
        """

    @abstractmethod
    def logmap(self, y, x, c):
        """Map a point y on the manifold to the tangent space of x.

        Args:
            y (tensor): A point on the manifold.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of mapping y to the tangent space of x.
        """

    @abstractmethod
    def logmap0(self, y, c):
        """Map a point y on the manifold to the tangent space of origin point.

        Args:
            y (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of mapping y to the tangent space of origin point.
        """

    @abstractmethod
    def ptransp(self, v, x, y, c):
        """Parallel transport function, used to move point v in the tangent space of x to the tangent space of y.

        Args:
            v (tensor): A point in the tangent space of x.
            x (tensor): A point on the manifold.
            y (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of transporting v from the tangent space at x to the tangent space at y.
        """

    @abstractmethod
    def ptransp0(self, v, x, c):
        """Parallel transport function, used to move point v in the tangent space of origin point to the tangent space of y.

        Args:
            v (tensor): A point in the tangent space of origin point.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of transporting v from the tangent space at origin point to the tangent space at y.
        """

    @abstractmethod
    def sqdist(self, x, y, c):
        """Calculate the squared geodesic/distance between x and y.

        Args:
            x (tensor): A point on the manifold.
            y (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: the squared geodesic/distance between x and y.
        """

    @abstractmethod
    def egrad2rgrad(self, grad, x, c):
        """Computes Riemannian gradient from the Euclidean gradient, typically used in Riemannian optimizers.

        Args:
            grad (tensor): Euclidean gradient at x.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: Riemannian gradient at x.
        """

    @abstractmethod
    def inner(self, v1, v2, x, c, keep_shape):
        """Computes the inner product of a pair of tangent vectors v1 and v2 at x.

        Args:
            v1 (tensor): A tangent point at x.
            v2 (tensor): A tangent point at x.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.
            keep_shape (bool, optional): Whether the output tensor keeps shape or not.

        Returns:
            tensor: The inner product of v1 and v2 at x.
        """

    def retraction(self, v, x, c):
        """Retraction is a continuous map function from tangent space to the manifold, typically used in Riemannian optimizers.
        The exp map is one of retraction functions.

        Args:
            v (tensor): A tangent point at x.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of mapping tangent point v at x to the manifold.
        """
        return self.proj(self.expmap(v, x, c), c)
