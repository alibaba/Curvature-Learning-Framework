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

"""A set of basic math functions used in constant curvature manifolds."""

import tensorflow as tf
from tensorflow import tan, atan, cos, acos, sin, asin
from tensorflow import tanh, atanh, cosh, acosh, sinh, asinh


def Tan(x):
    """Computes tangent of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor. Has the same type as x.
    """
    return tan(x)


def Tanh(x):
    """Computes hyperbolic tangent of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor: Has the same type as x.
    """
    return tanh(tf.clip_by_value(x, clip_value_min=-15, clip_value_max=15))


def TanC(x, c):
    """A unified tangent and inverse tangent function for different signs of curvatures.

    This function is used in k-Stereographic model, a unification of constant curvature manifolds.
    Please refer to https://arxiv.org/abs/2007.07698 for more details.

    First-order expansion is used in order to calculate gradients correctly when c is zero.

    Args:
        x (tensor): A tensor.
        c (float): Manifold curvature.

    Returns:
        A tensor: Has the same type of x.
    """
    return tf.cond(
        tf.math.abs(c) < 1e-5,
        lambda: x + 1 / 3 * (tf.math.sign(c) * c) * (x ** 3),  # 1-order taylor series at zero
        lambda: tf.cond(
            tf.math.sign(c) < 0.0,
            lambda: 1 / tf.math.sqrt(tf.math.abs(c)) * Tanh(x * tf.math.sqrt(tf.math.abs(c))),  # c < 0
            lambda: 1 / tf.math.sqrt(tf.math.abs(c)) * Tan(x * tf.math.sqrt(tf.math.abs(c)))  # c > 0
        )
    )


def ArTan(x):
    """Computes inverse tangent of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor: Has the same type as x.
    """
    return atan(tf.clip_by_value(x, clip_value_min=-1e15, clip_value_max=1e15))


def ArTanh(x):
    """Computes inverse hyperbolic tangent of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor: Has the same type as x.
    """
    return atanh(tf.clip_by_value(x, clip_value_min=-1 + 1e-7, clip_value_max=1 - 1e-7))


def ArTanC(x, c):
    """A unified hyperbolic tangent and inverse hyperbolic tangent function for different signs of curvatures.

    This function is used in k-Stereographic model, a unification of constant curvature manifolds.
    Please refer to https://arxiv.org/abs/2007.07698 for more details.

    First-order expansion is used in order to calculate gradients correctly when c is zero.

    Args:
        x (tensor): A tensor.
        c (float): Manifold curvature.

    Returns:
        A tensor: Has the same type of x.
    """
    return tf.cond(
        tf.math.abs(c) < 1e-5,
        lambda: x - 1 / 3 * (tf.math.sign(c) * c) * (x ** 3),  # 1-order taylor series at zero
        lambda: tf.cond(
            tf.math.sign(c) < 0.0,
            lambda: 1 / tf.math.sqrt(tf.math.abs(c)) * ArTanh(x * tf.math.sqrt(tf.math.abs(c))),  # c < 0
            lambda: 1 / tf.math.sqrt(tf.math.abs(c)) * ArTan(x * tf.math.sqrt(tf.math.abs(c)))  # c > 0
        )
    )


def Cos(x):
    """Computes cosine of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor: Has the same type of x.
    """
    return cos(x)


def Cosh(x):
    """Computes hyperbolic cosine of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor: Has the same type of x.
    """
    return cosh(tf.clip_by_value(x, clip_value_min=-15, clip_value_max=15))


def ArCos(x):
    """Computes inverse cosine of x element-wise.

    Args:
        x (tensor): A tensor.

    Returns:
        A tensor: Has the same type of x.
    """
    return acos(tf.clip_by_value(x, clip_value_min=-1 + 1e-7, clip_value_max=1 - 1e-7))


def ArCosh(x):
    """Computes inverse hyperbolic cosine of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor: Has the same type of x.
    """
    return acosh(tf.clip_by_value(x, clip_value_min=1 + 1e-7, clip_value_max=1e15))


def Sin(x):
    """Computes sine of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor: Has the same type of x.
    """
    return sin(x)


def Sinh(x):
    """Computes hyperbolic sine of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor: Has the same type of x.
    """
    return sinh(tf.clip_by_value(x, clip_value_min=-15, clip_value_max=15))


def SinC(x, c):
    """A unified sine and inverse sine function for different signs of curvatures.

    This function is used in k-Stereographic model, a unification of constant curvature manifolds.
    Please refer to https://arxiv.org/abs/2007.07698 for more details.

    First-order expansion is used in order to calculate gradients correctly when c is zero.

    Args:
        x (tensor): A tensor.
        c (float): Manifold curvature.
    
    Returns:
        A tensor: Has the same type of x.
    """
    return tf.cond(
        tf.math.abs(c) < 1e-5,
        lambda: x - 1 / 6 * (tf.math.sign(c) * c) * (x ** 3),  # 1-order taylor series at zero
        lambda: tf.cond(
            tf.math.sign(c) < 0.0,
            lambda: 1 / tf.math.sqrt(tf.math.abs(c)) * Sinh(x * tf.math.sqrt(tf.math.abs(c))),  # c < 0
            lambda: 1 / tf.math.sqrt(tf.math.abs(c)) * Sin(x * tf.math.sqrt(tf.math.abs(c)))  # c > 0
        )
    )


def ArSin(x):
    """Computes inverse sine of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor: Has the same type of x.
    """
    return asin(tf.clip_by_value(x, clip_value_min=-1 + 1e-7, clip_value_max=1 - 1e-7))


def ArSinh(x):
    """Computes inverse hyperbolic sine of x element-wise.
    
    Args:
        x (tensor): A tensor.
    
    Returns:
        A tensor: Has the same type of x.
    """
    return asinh(tf.clip_by_value(x, clip_value_min=-1e15, clip_value_max=1e15))


def ArSinC(x, c):
    """A unified hyperbolic sine and inverse hyperbolic sine function for different signs of curvatures.

    This function is used in k-Stereographic model, a unification of constant curvature manifolds.
    Please refer to https://arxiv.org/abs/2007.07698 for more details.

    First-order expansion is used in order to calculate gradients correctly when c is zero.

    Args:
        x (tensor): A tensor.
        c (float): Manifold curvature.

    Returns:
        A tensor: Has the same type of x.
    """
    return tf.cond(
        tf.math.abs(c) < 1e-5,
        lambda: x - 1 / 6 * (tf.math.sign(c) * c) * (x ** 3),  # 1-order taylor series at zero
        lambda: tf.cond(
            tf.math.sign(c) < 0.0,
            lambda: 1 / tf.math.sqrt(tf.math.abs(c)) * ArSinh(x * tf.math.sqrt(tf.math.abs(c))),  # c < 0
            lambda: 1 / tf.math.sqrt(tf.math.abs(c)) * ArSin(x * tf.math.sqrt(tf.math.abs(c)))  # c > 0
        )
    )
