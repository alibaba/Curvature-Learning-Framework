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

"""Euclidean manifold."""

from curvlearn.manifolds.manifold import tf, Manifold


class Euclidean(Manifold):
    """Euclidean Manifold class. Usually we refer it as R^n.

    Attributes:
        name (str): The manifold name, and its value is "Euclidean".
    """

    def __init__(self, **kwargs):
        """Initialize an Euclidean manifold.

        Args:
            **kwargs: Description
        """
        super(Euclidean, self).__init__(**kwargs)
        self.name = 'Euclidean'

    def proj(self, x, c):
        """A projection function that prevents x from leaving the manifold.

        Args:
            x (tensor): A point should be on the manifold, but it may not meet the manifold constraints.
            c (float): The manifold curvature.

        Returns:
            tensor: A projected point, meeting the manifold constraints.
        """
        return x

    def proj_tan(self, v, x, c):
        """A projection function that prevents v from leaving the tangent space of point x.

        Args:
            v (tensor): A point should be on the tangent space, but it may not meet the manifold constraints.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: A projected point, meeting the tangent space constraints.
        """
        return v

    def proj_tan0(self, v, c):
        """projection function that prevents v from leaving the tangent space of origin point.

        Args:
            v (tensor): A point should be on the tangent space, but it may not meet the manifold constraints.
            c (float): The manifold curvature.

        Returns:
            tensor: A projected point, meeting the tangent space constraints.
        """
        return v

    def expmap(self, v, x, c):
        """Map a point v in the tangent space of point x to the manifold.

        Args:
            v (tensor): A point in the tangent space of point x.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of mapping tangent point v to the manifold.
        """
        return v + x

    def expmap0(self, v, c):
        """Map a point v in the tangent space of origin point to the manifold.

        Args:
            v (tensor): A point in the tangent space of origin point.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of mapping tangent point v to the manifold.
        """
        return v

    def logmap(self, y, x, c):
        """Map a point y on the manifold to the tangent space of x.

        Args:
            y (tensor): A point on the manifold.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of mapping y to the tangent space of x.
        """
        return y - x

    def logmap0(self, y, c):
        """Map a point y on the manifold to the tangent space of origin point.

        Args:
            y (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of mapping y to the tangent space of origin point.
        """
        return y

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
        return tf.ones_like(x) * v

    def ptransp0(self, v, x, c):
        """Parallel transport function, used to move point v in the tangent space of origin point to the tangent space of y.

        Args:
            v (tensor): A point in the tangent space of origin point.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: The result of transporting v from the tangent space at origin point to the tangent space at y.
        """
        return tf.ones_like(x) * v

    def sqdist(self, x, y, c):
        """Calculate the squared geodesic/distance between x and y.

        Args:
            x (tensor): A point on the manifold.
            y (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: the squared geodesic/distance between x and y.
        """
        sqdis = tf.reduce_sum(tf.pow(x - y, 2), axis=-1)
        return sqdis

    def egrad2rgrad(self, grad, x, c):
        """Computes Riemannian gradient from the Euclidean gradient, typically used in Riemannian optimizers.

        Args:
            grad (tensor): Euclidean gradient at x.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.

        Returns:
            tensor: Riemannian gradient at x. Return the same value as grad in Euclidean manifold.
        """
        return grad

    def inner(self, v1, v2, x, c, keep_shape=False):
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
        if keep_shape:
            # In order to keep the same computation logic in Ada* Optimizer
            return v1 * v2
        else:
            return tf.reduce_sum(v1 * v2, axis=-1)
