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

"""kappa-Stereographic Manifold."""

from curvlearn.manifolds.manifold import tf, Manifold
from curvlearn.manifolds.math import TanC, ArTanC


class Stereographic(Manifold):
    """A Universal Model for Hyperbolic, Euclidean and Spherical Geometries.

    When c < 0, it maintains hyperbolic geometry.
    When c = 0, it maintains euclidean geometry.
    When c > 0, it maintains spherical geometry.
    Refer to https://andbloch.github.io/K-Stereographic-Model/ for intuition.
    Refer to https://arxiv.org/abs/1911.05076 for theory.
    
    Attributes:
        name (str): The manifold name. The value is "Stereographic".
        sum_method (str): The method to sum a list of embeddings. Default is einstein sum
    """

    def __init__(self, **kwargs):
        """Initialize a stereographic manifold.

        Args:
            name (str): The default name is "Stereographic".
            sum_method (str): The default method is "einstein".
        """
        super(Stereographic, self).__init__(**kwargs)
        self.name = 'Stereographic'

        self.sum_method = kwargs["sum_method"] if "sum_method" in kwargs else "einstein"
        self.truncate_c = lambda x: x

    def proj(self, x, c):
        """A projection function that prevents x from leaving the manifold.
        
        Args:
            x (tensor): A point should be on the manifold, but it may not meet the manifold constraints.
            c (float): The manifold curvature.
        
        Returns:
            tensor: A projected point, meeting the manifold constraints.
        """
        c = self.truncate_c(c)
        return tf.cond(
            c < tf.constant(0.0, dtype=self.dtype),
            lambda: tf.clip_by_norm(
                x, (1.0 - self.eps) / tf.sqrt(-c), axes=-1),
            lambda: x
        )

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
        """A projection function that prevents v from leaving the tangent space of origin point.
        
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
        c = self.truncate_c(c)
        v_norm = self.clip(tf.norm(v, ord=2, axis=-1, keepdims=True))
        second_term = TanC(self._lambda_x(x, c) * v_norm / 2.0, c) * v / v_norm
        gamma = self._mobius_add(x, second_term, c)
        return gamma

    def expmap0(self, v, c):
        """Map a point v in the tangent space of origin point to the manifold.
        
        Args:
            v (tensor): A point in the tangent space of origin point.
            c (float): The manifold curvature.
        
        Returns:
            tensor: The result of mapping tangent point v to the manifold.
        """
        c = self.truncate_c(c)
        v_norm = self.clip(tf.norm(v, ord=2, axis=-1, keepdims=True))
        gamma = TanC(v_norm, c) * v / v_norm
        return gamma

    def logmap(self, y, x, c):
        """Map a point y on the manifold to the tangent space of x.
        
        Args:
            y (tensor): A point on the manifold.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.
        
        Returns:
            tensor: The result of mapping y to the tangent space of x.
        """
        c = self.truncate_c(c)
        sub = self._mobius_add(-x, y, c)
        sub_norm = self.clip(tf.norm(sub, ord=2, axis=-1, keepdims=True))
        lam = self._lambda_x(x, c)
        return 2.0 / lam * ArTanC(sub_norm, c) * sub / sub_norm

    def logmap0(self, y, c):
        """Map a point y on the manifold to the tangent space of origin point.
        
        Args:
            y (tensor): A point on the manifold.
            c (float): The manifold curvature.
        
        Returns:
            tensor: The result of mapping y to the tangent space of origin point.
        """
        c = self.truncate_c(c)
        y_norm = self.clip(tf.norm(y, ord=2, axis=-1, keepdims=True))
        return ArTanC(y_norm, c) * y / y_norm

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
        c = self.truncate_c(c)
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, v, c) * lambda_x / lambda_y

    def ptransp0(self, v, x, c):
        """Parallel transport function, used to move point v in the tangent space of origin point to the tangent space of y.
        
        Args:
            v (tensor): A point in the tangent space of origin point.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.
        
        Returns:
            tensor: The result of transporting v from the tangent space at origin point to the tangent space at y.
        """
        c = self.truncate_c(c)
        lambda_x = self._lambda_x(x, c)
        return tf.constant([2.0], dtype=self.dtype) * v / lambda_x

    def sqdist(self, x, y, c):
        """Calculate the squared geodesic/distance between x and y.
        
        Args:
            x (tensor): A point on the manifold.
            y (tensor): A point on the manifold.
            c (float): The manifold curvature.
        
        Returns:
            tensor: the squared geodesic/distance between x and y.
        """
        c = self.truncate_c(c)
        dist = 2.0 * ArTanC(tf.norm(self._mobius_add(-x, y, c), ord=2, axis=-1), c)
        return dist ** 2

    def egrad2rgrad(self, grad, x, c):
        """Computes Riemannian gradient from the Euclidean gradient, typically used in Riemannian optimizers.
        
        Args:
            grad (tensor): Euclidean gradient at x.
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.
        
        Returns:
            tensor: Riemannian gradient at x.
        """
        c = self.truncate_c(c)
        metric = tf.square(self._lambda_x(x, c))
        return grad / metric

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
        c = self.truncate_c(c)
        metric = tf.square(self._lambda_x(x, c))
        product = v1 * metric * v2
        res = tf.reduce_sum(product, axis=-1, keepdims=True)
        if keep_shape:
            # return tf.broadcast_to(res, x.shape)
            last_dim = x.shape.as_list()[-1]
            return tf.concat([res for _ in range(last_dim)], axis=-1)
        return tf.squeeze(res, axis=-1)

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
        c = self.truncate_c(c)
        new_v = self.expmap(v, x, c)
        return self.proj(new_v, c)

    def _mobius_add(self, x, y, c):
        """Möbius vector addition.

        The generalized vector addition in möbius gyrovector space, which is a noncommutative and non-associative binary operation.
        While curvature c -> 0, it converges to vector addition in Euclidean space, the curvature of which is zero.

        Args:
            x (tensor): A point on the manifold.
            y (tensor): A point on the manifold.
            c (float): The manifold curvature.
        
        Returns:
            tensor: The result of adding x and y.
        """
        c = self.truncate_c(c)
        x2 = tf.reduce_sum(tf.pow(x, 2), axis=-1, keepdims=True)
        y2 = tf.reduce_sum(tf.pow(y, 2), axis=-1, keepdims=True)
        xy = tf.reduce_sum(x * y, axis=-1, keepdims=True)
        num = (1 - 2 * c * xy - c * y2) * x + (1 + c * x2) * y
        denom = 1 - 2 * c * xy + (c ** 2) * x2 * y2
        return num / self.clip(denom)

    def _mobius_mul(self, x, a, c):
        """Möbius scalar multiplication.

        While curvature c -> 0, it converges to Euclidean scalar multiplication.

        Args:
            x (tensor): A point on the manifold.
            a (float): A scalar.
            c (float): The manifold curvature.
        
        Returns:
            tensor: The result of a * x in möbius gyrovector space.
        """
        c = self.truncate_c(c)
        x_norm = self.clip(tf.norm(x, ord=2, axis=-1, keepdims=True))
        scale = TanC(a * ArTanC(x_norm, c), c) / x_norm
        return scale * x

    def _mobius_matvec(self, x, a, c):
        c = self.truncate_c(c)
        x_norm = self.clip(tf.norm(x, ord=2, axis=-1, keepdims=True))
        mx = tf.matmul(x, a)
        mx_norm = self.clip(tf.norm(mx, ord=2, axis=-1, keepdims=True))
        res = TanC(mx_norm / x_norm * ArTanC(x_norm, c), c) * mx / mx_norm
        return res

    def _lambda_x(self, x, c):
        """A conformal factor.
        
        Args:
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.
        
        Returns:
            tensor: the conformal factor of x.
        """
        c = self.truncate_c(c)
        x_sqnorm = tf.reduce_sum(tf.pow(x, 2), axis=-1, keepdims=True)
        return self.clip(2.0 / (1.0 + c * x_sqnorm))

    def _gyration(self, x, y, v, c):
        """With a groupoid (G, oplus) consisting of a set G and a certain binary operation oplus,
        a gyration is a operator gry[x,y] that generates an automorphism G -> G given by z mapsto gry[x,y]z.
        
        Args:
            x (tensor): A point on the manifold.
            y (tensor): A point on the manifold.
            v (tensor): A point on the manifold.
            c (float): The manifold curvature.
        
        Returns:
            tensor: gyr[x, y]z = -(x + y) + (x + (y + z)).
        """
        xy = self._mobius_add(x, y, c)
        yv = self._mobius_add(y, v, c)
        xyv = self._mobius_add(x, yv, c)
        return self._mobius_add(-xy, xyv, c)

    def _antipode(self, x, c):
        """Computes the antipode of x.
        
        Args:
            x (tensor): A point on the manifold.
            c (float): The manifold curvature.
        
        Returns:
            tensor: The antipode of x.
        """
        lambda_x = self._lambda_x(x, c)
        x_sqnorm = tf.reduce_sum(tf.pow(x, 2), axis=-1, keepdims=True)
        scale = -1.0 / (lambda_x * c * x_sqnorm)
        return scale * x

    def weight_sum(self, t, a, c=None):
        """Sums tensor list t by weight list a.

        Args:
            t (list): A list of tensors. The shape of each tensor is [batch, ..., dim]
            a (list): A list of tensors as the weights. The shape of each tensor is [batch, ..., 1]
            c (float, optional): The manifold curvature.

        Returns:
            tensor: The weighted sum result. The shape is [batch, ..., dim].
        """
        c = self.truncate_c(c)
        if not self.sum_method == "einstein":
            return super(Stereographic, self).weight_sum(t, a, c)

        x, a = tf.stack(t, axis=0), tf.stack(a, axis=0)

        lambda_x = self._lambda_x(x, c)
        denom = tf.reduce_sum((lambda_x - 1.0) * a, axis=0)
        nom = tf.reduce_sum(lambda_x * a * x, axis=0)
        mc = self._mobius_mul(nom / denom, 0.5, c)

        def get_better_dis():
            mca = self._antipode(mc, c)
            _mc = tf.stack([mc for _ in range(x.shape.as_list()[0])], axis=0)
            _mca = tf.stack([mca for _ in range(x.shape.as_list()[0])], axis=0)
            mc_dis = tf.reduce_sum(
                a * tf.expand_dims(self.sqdist(_mc, x, c), axis=-1), axis=0)
            mca_dis = tf.reduce_sum(
                a * tf.expand_dims(self.sqdist(_mca, x, c), axis=-1), axis=0)
            return tf.where(
                tf.squeeze(mc_dis < mca_dis),
                mc,
                mca
            )

        weight_sum = tf.cond(
            c > tf.constant(0.0, dtype=self.dtype),
            get_better_dis,
            lambda: mc
        )

        scale = tf.reduce_sum(a, axis=0)
        return self._mobius_mul(weight_sum, scale, c)


class PoincareBall(Stereographic):
    """PoincareBall Manifold class.

    We have x1^2 + x1^2 + ... + xn^2 < -1 / c. (c < 0)
    So that the radius will be 1 / sqrt(-c).
    Notice that the more close c is to 0, the more flat space will be.

    Curvature(c) in PoincareBall will be truncated to negative, whereas no limitation in Stereographic.
    
    Attributes:
        name (str): The manifold name. The value is "Stereographic".
    """

    def __init__(self, **kwargs):
        super(PoincareBall, self).__init__(**kwargs)
        self.name = 'PoincareBall'

        self.truncate_c = lambda x: tf.clip_by_value(x, clip_value_min=tf.float32.min, clip_value_max=-1e-5)


class ProjectedSphere(Stereographic):
    """Projected Sphere Manifold class.

    Curvature(c) in ProjectedSphere will be truncated to positive, whereas no limitation in Stereographic.
    
    Attributes:
        default_c (float): The manifold curvature.
        name (str): The manifold name.
    """

    def __init__(self, **kwargs):
        super(ProjectedSphere, self).__init__(**kwargs)
        self.name = 'ProjectedSphere'

        self.truncate_c = lambda x: tf.clip_by_value(x, clip_value_min=1e-5, clip_value_max=tf.float32.max)
