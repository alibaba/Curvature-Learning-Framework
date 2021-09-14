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

"""A set of loss functions"""

import tensorflow as tf


class Loss(object):
    """A set of loss functions
    """
    
    def __init__(self, name, weight_decay=0.0):
        """Initialization.
        
        Args:
            name (str): The name of loss function.
            weight_decay (float, optional): The factor for regularization term.
        
        Raises:
            NotImplementedError: Loss function not implemented.
        """
        super(Loss, self).__init__()

        self.weight_decay = weight_decay

        if hasattr(self, name):
            self.loss = getattr(self, name)
        else:
            raise NotImplementedError

    def softmax_loss(self, sim, ratio=1.0):
        """Computes the softmax loss.
        
        Args:
            sim (tensor): The sim value for one positive sample and several negative samples.
            ratio (float, optional): The scale factor.
        
        Returns:
            tensor: The softmax loss
        """
        prob = tf.nn.softmax(ratio * sim)
        hit_prob = tf.slice(prob, [0, 0], [-1, 1])
        loss = -tf.log(hit_prob)

        return tf.reduce_mean(loss, name='softmax_loss')

    def bce_loss(self, sim):
        """Computes the bce (binary cross entropy) loss.
        
        Args:
            sim (tensor): The sim value for one positive sample and several negative samples.
        
        Returns:
            tensor: The bce loss.
        """
        # bce_loss = -log(sigmoid(sim^+)) + -log(1-sigmoid(sim^-))
        hit = tf.slice(sim, [0, 0], [-1, 1])
        miss = tf.slice(sim, [0, 1], [-1, -1])
        labels = tf.concat([tf.ones_like(hit), tf.zeros_like(miss)], axis=-1)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=sim
        )

        return tf.reduce_mean(loss, name='bce_loss')

    def triplet_loss(self, sim, margin=0.5):
        """Computes the triplet loss.
        
        Args:
            sim (tensor): The sim value for one positive sample and several negative samples.
            margin (float, optional): The margin value.
        
        Returns:
            tensor: The triplet loss.
        """
        # sim = sigmoid(5.0 - 5.0*distance)
        pos_sim, neg_sim = tf.slice(sim, [0, 0], [-1, 1]), tf.slice(sim, [0, 1], [-1, -1])
        # pos_sim has larger similarity than neg_sim
        triplet_loss = margin + neg_sim - pos_sim
        triplet_loss = tf.nn.relu(triplet_loss)
        triplet_loss = tf.reduce_sum(triplet_loss, axis=-1)
        triplet_loss = tf.reduce_mean(triplet_loss)

        # wmrb_loss = tf.log(1.0 + margin_loss)

        return triplet_loss

    def ranking_loss(self, sim, margin=1.0):
        """Computes the ranking loss.
        
        Args:
            sim (tensor): The sim value for one positive sample and several negative samples.
            margin (float, optional): The margin value.
        
        Returns:
            tensor: The ranking loss.
        """
        # sim = 1.0 - distance
        pos_sim, neg_sim = tf.slice(sim, [0, 0], [-1, 1]), tf.slice(sim, [0, 1], [-1, -1])
        pos_dis, neg_dis = 1.0 - pos_sim, 1.0 - neg_sim
        hinge_loss = tf.nn.relu(margin - neg_dis)
        ranking_loss = tf.reduce_mean(
            pos_dis) + tf.reduce_mean(tf.reduce_sum(hinge_loss, axis=-1))

        return ranking_loss

    def bpr_loss(self, sim):
        """Computes the bpr loss.
        
        Args:
            sim (tensor): The sim value for one positive sample and several negative samples.
        
        Returns:
            tensor: The bpr loss.
        """
        # sim = 1.0 - distance
        pos_sim, neg_sim = tf.slice(sim, [0, 0], [-1, 1]), tf.slice(sim, [0, 1], [-1, -1])
        margin = pos_sim - neg_sim
        # bpr loss = -log(sigmoid(x))
        labels = tf.ones_like(margin)
        bpr_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=margin
        )

        return tf.reduce_mean(bpr_loss, name='bpr_loss')

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)
        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # return self.loss(*args, **kwargs) + self.weight_decay * tf.add_n(reg_losses)
