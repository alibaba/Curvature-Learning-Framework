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

"""Product manifold."""

from curvlearn.manifolds.manifold import tf, Manifold


class Product(Manifold):
    """Product Manifold, served as a composition of multi-manifolds.
    
    Usage:
            Product_Manifold = Product((Manifold_1,dim_1),(Manifold_2,dim_2),...)
            or
            Product_Manifold = Product(Manifold_1,Manifold_2,...)
        The latter will split input dimension equivalently.
        Notice the curvature of Product manifold is a list of sub-manifolds' curvatures.
    
    Attributes:
        attn (bool): Whether use attention or not, default is False.
        fusing (bool): Whether use fusing or not, default is False.
        name (str): The manifold name.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a Product manifold.
        """
        super(Product, self).__init__(**kwargs)
        self.name = 'Product'

        self.sub_manifolds, self.sub_dims = [], []
        for manifold_config in args:
            if type(manifold_config) is tuple:
                self.sub_manifolds.append(manifold_config[0])
                self.sub_dims.append(int(manifold_config[1]))
            else:
                self.sub_manifolds.append(manifold_config)
                self.sub_dims.append(int(1))

        assert all([isinstance(man, Manifold) for man in self.sub_manifolds]), "Fail to Build Product Manifold"

        self.fusing = kwargs["fusing"] if "fusing" in kwargs else False
        self.attn = kwargs["attn"] if "attn" in kwargs else False

    def split_by_manifold(self, t):
        """Splits a tensor (or tensor list) into a list of sub manifolds.
        
        Args:
            t (tensor or list): A tensor or a list of tensors.
        
        Returns:
            list: A list of tensors. Each tensor represents a sub manifold.
        """
        if type(t) is list:
            return t
        feature_dim = t.shape.as_list()[-1]
        manifold_dim_ratio = feature_dim // sum(self.sub_dims)
        manifold_dim = [manifold_dim_ratio * dim for dim in self.sub_dims]
        return tf.split(t, num_or_size_splits=manifold_dim, axis=-1)

    @staticmethod
    def split_curvature(c):
        """Splits a tensor (or tensor list) into a list of sub curvatures.

        Args:
            c (tensor or list): A tensor or a list of tensors.

        Returns:
            list: Contains a list of tensors. Each tensor represents a sub curvature.
        """
        if type(c) is list:
            return c
        return tf.unstack(c, axis=0)

    @staticmethod
    def combine_by_manifold(tensor_list, as_list=False):
        """Combines a list of sub manifolds.
        
        Args:
            tensor_list (tensor or list): A tensor or a list of tensors.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor or list: Returns a single tensor or a list of sub tensors.
        """
        if as_list:
            return tensor_list
        return tf.concat(tensor_list, axis=-1)

    def proj(self, x, c, as_list=False):
        """A projection function that prevents x from leaving the manifold.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.
        
        Args:
            x (tensor, list): A point should be on the manifold, but it may not meet the manifold constraints.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: A projected point, meeting the manifold constraints.
        """
        sub_c = self.split_curvature(c)
        sub_x = self.split_by_manifold(x)
        proj_list = [man.proj(_x, _c) for man, _x, _c in zip(self.sub_manifolds, sub_x, sub_c)]
        return self.combine_by_manifold(proj_list, as_list=as_list)

    def proj_tan(self, v, x, c, as_list=False):
        """A projection function that prevents v from leaving the tangent space of point x.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            v (tensor, list): A point should be on the tangent space, but it may not meet the manifold constraints.
            x (tensor, list): A point on the manifold.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: A projected point, meeting the tangent space constraints.
        """
        sub_c = self.split_curvature(c)
        sub_v, sub_x = self.split_by_manifold(v), self.split_by_manifold(x)
        proj_tan_list = [man.proj_tan(_v, _x, _c) for man, _v, _x, _c in zip(self.sub_manifolds, sub_v, sub_x, sub_c)]
        return self.combine_by_manifold(proj_tan_list, as_list=as_list)

    def proj_tan0(self, v, c, as_list=False):
        """A projection function that prevents v from leaving the tangent space of origin point.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            v (tensor, list): A point should be on the tangent space, but it may not meet the manifold constraints.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: A projected point, meeting the tangent space constraints.
        """
        sub_c = self.split_curvature(c)
        sub_v = self.split_by_manifold(v)
        proj_tan0_list = [man.proj_tan0(_v, _c) for man, _v, _c in zip(self.sub_manifolds, sub_v, sub_c)]
        return self.combine_by_manifold(proj_tan0_list, as_list=as_list)

    def expmap(self, v, x, c, as_list=False):
        """Map a point v in the tangent space of point x to the manifold.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            v (tensor, list): A point in the tangent space of point x.
            x (tensor, list): A point on the manifold.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: The result of mapping tangent point v to the manifold.
        """
        sub_c = self.split_curvature(c)
        sub_v, sub_x = self.split_by_manifold(v), self.split_by_manifold(x)
        expmap_list = [man.expmap(_v, _x, _c) for man, _v, _x, _c in zip(self.sub_manifolds, sub_v, sub_x, sub_c)]
        return self.combine_by_manifold(expmap_list, as_list=as_list)

    def expmap0(self, v, c, as_list=False):
        """Map a point v in the tangent space of origin point to the manifold.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            v (tensor, list): A point in the tangent space of origin point.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: The result of mapping tangent point v to the manifold.
        """
        sub_c = self.split_curvature(c)
        sub_v = self.split_by_manifold(v)
        expmap0_list = [man.expmap0(_v, _c) for man, _v, _c in zip(self.sub_manifolds, sub_v, sub_c)]
        return self.combine_by_manifold(expmap0_list, as_list=as_list)

    def logmap(self, y, x, c, as_list=False):
        """ap a point y on the manifold to the tangent space of x.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            y (tensor, list): A point on the manifold.
            x (tensor, list): A point on the manifold.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: The result of mapping y to the tangent space of x.
        """
        sub_c = self.split_curvature(c)
        sub_y, sub_x = self.split_by_manifold(y), self.split_by_manifold(x)
        logmap_list = [man.logmap(_y, _x, _c) for man, _y, _x, _c in zip(self.sub_manifolds, sub_y, sub_x, sub_c)]
        return self.combine_by_manifold(logmap_list, as_list=as_list)

    def logmap0(self, y, c, as_list=False):
        """Map a point y on the manifold to the tangent space of origin point.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            y (tensor, list): A point on the manifold.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: The result of mapping y to the tangent space of origin point.
        """
        sub_c = self.split_curvature(c)
        sub_y = self.split_by_manifold(y)
        logmap0_list = [man.logmap0(_y, _c) for man, _y, _c in zip(self.sub_manifolds, sub_y, sub_c)]
        return self.combine_by_manifold(logmap0_list, as_list=as_list)

    def ptransp(self, v, x, y, c, as_list=False):
        """Parallel transport function, used to move point v in the tangent space of x to the tangent space of y.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            v (tensor, list): A point in the tangent space of x.
            x (tensor, list): A point on the manifold.
            y (tensor, list): A point on the manifold.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: The result of transporting v from the tangent space at x to the tangent space at y.
        """
        sub_c = self.split_curvature(c)
        sub_v = self.split_by_manifold(v)
        sub_x, sub_y = self.split_by_manifold(x), self.split_by_manifold(y)
        ptransp_list = [man.ptransp(_v, _x, _y, _c) for man, _v, _x, _y, _c in
                        zip(self.sub_manifolds, sub_v, sub_x, sub_y, sub_c)]
        return self.combine_by_manifold(ptransp_list, as_list=as_list)

    def ptransp0(self, v, x, c, as_list=False):
        """Parallel transport function, used to move point v in the tangent space of origin point to the tangent space of y.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            v (tensor, list): A point in the tangent space of origin point.
            x (tensor, list): A point on the manifold.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: The result of transporting v from the tangent space at origin point to the tangent space at y.
        """
        sub_c = self.split_curvature(c)
        sub_v, sub_x = self.split_by_manifold(v), self.split_by_manifold(x)
        ptransp0_list = [man.ptransp0(_v, _x, _c) for man, _v, _x, _c in zip(self.sub_manifolds, sub_v, sub_x, sub_c)]
        return self.combine_by_manifold(ptransp0_list, as_list=as_list)

    def sqdist(self, x, y, c, as_list=False):
        """Calculate the squared geodesic/distance between x and y.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            x (tensor, list): A point on the manifold.
            y (tensor, list): A point on the manifold.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: the squared geodesic/distance between x and y.
        """
        sub_c = self.split_curvature(c)
        sub_x, sub_y = self.split_by_manifold(x), self.split_by_manifold(y)
        sqdist_list = [man.sqdist(_x, _y, _c) for man, _x, _y, _c in zip(self.sub_manifolds, sub_x, sub_y, sub_c)]
        sqdist_list = [tf.expand_dims(dist, -1) for dist in sqdist_list]
        return self.combine_by_manifold(sqdist_list, as_list=as_list)

    def egrad2rgrad(self, grad, x, c, as_list=False):
        """Computes Riemannian gradient from the Euclidean gradient, typically used in Riemannian optimizers.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            grad (tensor, list): Euclidean gradient at x.
            x (tensor, list): A point on the manifold.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: Riemannian gradient at x.
        """
        sub_c = self.split_curvature(c)
        sub_grad, sub_x = self.split_by_manifold(grad), self.split_by_manifold(x)
        egrad2rgrad_list = [man.egrad2rgrad(_grad, _x, _c) for man, _grad, _x, _c in
                            zip(self.sub_manifolds, sub_grad, sub_x, sub_c)]
        return self.combine_by_manifold(egrad2rgrad_list, as_list=as_list)

    def inner(self, v1, v2, x, c, keep_shape=False, as_list=False):
        """Computes the inner product of a pair of tangent vectors v1 and v2 at x.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            v1 (tensor, list): A tangent point at x.
            v2 (tensor, list): A tangent point at x.
            x (tensor, list): A point on the manifold.
            c (tensor, list): The manifold curvature.
            keep_shape (bool, optional): Whether the output tensor keeps shape or not.
            as_list (bool, optional): Whether the input is a list or not.

        Returns:
            tensor: The inner product of v1 and v2 at x.

        Raises:
            NotImplementedError: Description
        """
        sub_c = self.split_curvature(c)
        if keep_shape is False:
            # have different logic between Euclidean and Hyperbolic
            raise NotImplementedError
        sub_v1, sub_v2 = self.split_by_manifold(v1), self.split_by_manifold(v2)
        sub_x = self.split_by_manifold(x)
        inner_list = [man.inner(_v1, _v2, _x, _c, keep_shape=True)
                      for man, _v1, _v2, _x, _c in zip(self.sub_manifolds, sub_v1, sub_v2, sub_x, sub_c)]
        return self.combine_by_manifold(inner_list, as_list=as_list)

    def retraction(self, v, x, c, as_list=False):
        """Retraction is a continuous map function from tangent space to the manifold, typically used in Riemannian optimizers.
        The exp map is one of retraction functions.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            v (tensor, list): A tangent point at x.
            x (tensor, list): A point on the manifold.
            c (tensor, list): The manifold curvature.
            as_list (bool, optional): Whether the output is a list or not.
        
        Returns:
            tensor: The result of mapping tangent point v at x to the manifold.
        """
        sub_c = self.split_curvature(c)
        sub_v, sub_x = self.split_by_manifold(v), self.split_by_manifold(x)
        retraction_list = [man.retraction(_v, _x, _c) for man, _v, _x, _c in
                           zip(self.sub_manifolds, sub_v, sub_x, sub_c)]
        return self.combine_by_manifold(retraction_list, as_list=as_list)

    '''
    Notice ops in Product Manifold usually following such a pipeline: 
    preprocess by split -> single manifold op -> postprocess by concat
    Classic neural network ops are constructed by op stacking.
    A simple accelerating method is avoiding useless concat/split operations.
    '''

    def variable(self, t, c):
        """Defines a riemannian variable from manifold or tangent space at origin according to its name.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            t (tensor): A variable, can be in Euclidean space or a specific manifold.
            c (tensor): The manifold curvature.
        
        Returns:
            tensor: Variable on specific manifold. Has the same type with t.
        """
        c = self.split_curvature(c)

        if "RiemannianParameter" not in t.name:
            t = self.split_by_manifold(t)
            t = self.proj_tan0(t, c, as_list=True)
            t = self.expmap0(t, c, as_list=True)
        t = self.proj(t, c)

        return t

    def to_manifold(self, t, c, base=None):
        """Converts a tensor t in the tangent space of base point to the manifold.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            t (tensor): A tensor, should lie in Euclidean space.
            c (tensor): The manifold curvature.
            base (tensor, optional): A base point on the manifold.

        Returns:
            tensor: A tensor in the manifold. Has the same type as t.
        """
        c = self.split_curvature(c)
        t = self.split_by_manifold(t)

        if base is None:
            t = self.proj_tan0(t, c, as_list=True)
            manifold_t = self.expmap0(t, c, as_list=True)
        else:
            base = self.split_by_manifold(base)
            t = self.proj_tan(t, base, c, as_list=True)
            manifold_t = self.expmap(t, base, c, as_list=True)

        manifold_t = self.proj(manifold_t, c)

        return manifold_t

    # no need to simplify 'to_tangent' function
    # no need to simplify 'mean' function
    # no need to simplify 'sum' function

    '''
    Ops in Neural Network may involve complicated interaction between different manifolds,
    such as multiplication. Directly operating right-multiply means sum all results from sub-manifolds.
    '''

    def weight_sum(self, t, a, c):
        """Computes the sum of tensor list t with weight list a

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            t (list): A list of tensors. The shape of each tensor is [batch, ..., dim]
            a (list): A list of tensors as the weights. The shape of each tensor is [batch, ..., 1]
            c (tensor): The manifold curvature.

        Returns:
            tensor: The weighted sum result. The shape is [batch, ..., dim].
        """
        sub_c = self.split_curvature(c)

        tensor_list = [self.split_by_manifold(_t) for _t in t]
        tensor_list = list(zip(*tensor_list))
        tensor_list = [man.weight_sum(_x, a, c=_c) for man, _x, _c in zip(self.sub_manifolds, tensor_list, sub_c)]
        return self.combine_by_manifold(tensor_list)

    def concat(self, tensor_list, c, axis=-1):
        """Concatenates tensors along one dimension.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            tensor_list (list): A list of Tensor objects.
            c (tensor): The manifold curvature.
            axis (int, optional): Dimension along which to concatenate.

        Returns:
            tensor: A tensor resulting from concatenation of the input tensors. Lies in the same manifold as t.
        """
        sub_c = self.split_curvature(c)

        new_list = []
        for t in tensor_list:
            sub_t = self.split_by_manifold(t)
            new_list.append([man.logmap0(_t, _c) for man, _t, _c in zip(self.sub_manifolds, sub_t, sub_c)])
        tensor_list = [tf.concat(sub_manifold, axis=axis) for sub_manifold in zip(*new_list)]
        concat_tensor = tf.concat(tensor_list, axis=axis)
        concat_tensor = self.to_manifold(concat_tensor, c=c, base=None)
        return concat_tensor

    def linear(self, t, in_dim, out_dim, c_in, c_out, act, scope="linear"):
        """Computes the linear transformation and activation for the input tensor t.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            t (tensor): A tensor.
            in_dim (int): The dimension of the input tensor.
            out_dim (int): The dimension of the output tensor.
            c_in (tensor): The manifold curvature for the input tensor.
            c_out (tensor): The manifold curvature for the output tensor.
            act (function): The non-linear activation function.
            scope (str, optional): the scope name for the variable.

        Returns:
            tensor: The result tensor after linear transformation and activation.
        """
        if self.fusing:
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                t = self.fuse(t, in_dim, out_dim, c_in, c_out, act)
            return t

        sub_c_in, sub_c_out = self.split_curvature(c_in), self.split_curvature(c_out)
        sub_x = self.split_by_manifold(t)

        in_dim_ratio = in_dim // sum(self.sub_dims)
        out_dim_ratio = out_dim // sum(self.sub_dims)
        sub_in_dim = [in_dim_ratio * dim for dim in self.sub_dims]
        sub_out_dim = [out_dim_ratio * dim for dim in self.sub_dims]

        linear_list = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for idx, (man, _x, _in_dim, _out_dim, _c_in, _c_out) in enumerate(
                    zip(self.sub_manifolds, sub_x, sub_in_dim, sub_out_dim, sub_c_in, sub_c_out)):
                linear_list.append(
                    man.linear(_x, _in_dim, _out_dim, _c_in, _c_out, act=act, scope="product_linear_" + str(idx)))
            out = self.combine_by_manifold(linear_list)

        return out

    def matmul(self, t, m, c):
        """Multiplies tensor t by euclidean matrix m.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            t (tensor): A tensor.
            m (tensor): The parameter matrix, should lie in Euclidean.
            c (tensor): The manifold curvature.

        Returns:
            tensor: A tensor. Lies in the same manifold as t.
        """
        x = self.to_tangent(t, c=c, base=None)

        in_feature_dim = t.shape.as_list()[-1]
        out_feature_dim = m.shape.as_list()[-1]
        in_dim_ratio = in_feature_dim // sum(self.sub_dims)
        out_dim_ratio = out_feature_dim // sum(self.sub_dims)
        in_dim = [in_dim_ratio * dim for dim in self.sub_dims]
        out_dim = [out_dim_ratio * dim for dim in self.sub_dims]

        mask = []
        for i, d in enumerate(in_dim):
            mask.append(
                tf.concat(
                    [
                        tf.zeros([d, sum(out_dim[:i])]),
                        tf.ones([d, out_dim[i]]),
                        tf.zeros([d, sum(out_dim[i + 1:])])
                    ],
                    axis=-1
                )
            )
        mask = tf.concat(mask, axis=0)
        m = tf.multiply(mask, m)

        mx = tf.matmul(x, m)
        mx = self.to_manifold(mx, c=c, base=None)
        return mx

    def distance(self, src, tar, c):
        """Computes the squared geodesic/distance between src and tar.

        This function first splits the inputs into multiple variables with each associated to a sub manifold,
        then performs the corresponding operation in each sub manifold,
        and finally combines the results for each sub manifold.

        Args:
            src (tensor): The source point.
            tar (tensor): The target point. Lies in the same manifold as src.
            c (tensor): The manifold curvature.

        Returns:
            tensor: The distance between src and tar.
        """
        src, tar = self.proj(src, c), self.proj(tar, c)

        manifold_dis = self.sqdist(src, tar, c=c)

        if self.attn:
            weight_src, weight_tar = src, tar

            with tf.variable_scope("distance", reuse=tf.AUTO_REUSE):
                attn_weight = tf.get_variable(
                    "attn_weight",
                    shape=[weight_src.shape.as_list()[-1], len(self.sub_manifolds)],
                    dtype=self.dtype,
                    initializer=tf.glorot_normal_initializer(dtype=self.dtype)
                )

            src_weight = tf.matmul(weight_src, attn_weight)
            tar_weight = tf.matmul(weight_tar, attn_weight)

            src_weight = len(self.sub_manifolds) * tf.nn.softmax(src_weight, axis=-1)
            tar_weight = len(self.sub_manifolds) * tf.nn.softmax(tar_weight, axis=-1)

        else:
            src_weight = tf.ones_like(manifold_dis)
            tar_weight = tf.ones_like(manifold_dis)

        attn = (src_weight + tar_weight) / 2
        total_dis = tf.multiply(manifold_dis, attn)
        return tf.reduce_sum(total_dis, axis=-1, keepdims=True)

    def fuse(self, x, in_dim, out_dim, c_in, c_out, act):
        """Fuses multiple sub manifold embeddings.
        
        Args:
            x (tensor): A tensor or a list of tensors.
            in_dim (tensor): The dimension of the input tensor.
            out_dim (TYPE): The dimension of the output tensor.
            c_in (tensor): The manifold curvature for the input tensor.
            c_out (tensor): The manifold curvature for the output tensor.
            act (function): The non-linear activation function.
        
        Returns:
            tensor: The fused embeddings.
        """
        sub_c_in, sub_c_out = self.split_curvature(c_in), self.split_curvature(c_out)
        sub_x = self.split_by_manifold(x)

        in_dim_ratio = in_dim // sum(self.sub_dims)
        out_dim_ratio = out_dim // sum(self.sub_dims)
        sub_in_dim = [2 * in_dim_ratio * dim for dim in self.sub_dims]
        sub_out_dim = [out_dim_ratio * dim for dim in self.sub_dims]

        tangent_x = self.logmap0(sub_x, sub_c_in, as_list=True)
        aggreate_info = tf.reduce_mean(tf.stack(tangent_x, axis=0), axis=0)

        fused = []
        for idx, (man, _x, _in_dim, _out_dim, _c_in, _c_out) in enumerate(
                zip(self.sub_manifolds, tangent_x, sub_in_dim, sub_out_dim, sub_c_in, sub_c_out)):
            with tf.variable_scope("fused_manifold_" + str(idx), reuse=tf.AUTO_REUSE):
                new_x = tf.concat([_x, aggreate_info], axis=-1)
                new_x = man.to_manifold(new_x, c=_c_in, base=None)
                fused_m = man.linear(
                    new_x, _in_dim, _out_dim, _c_in, _c_out, act=act, scope="product_linear_" + str(idx))
            fused.append(fused_m)

        return self.combine_by_manifold(fused)
