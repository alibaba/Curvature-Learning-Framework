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

"""Learning rate warm up"""

from tensorflow.python.framework import ops
import tensorflow as tf


class LearningRate(object):
    """Gradually warm-up(increasing and decreasing) learning rate in optimizer.

    Includes three stages: warm up stage, increasing stage and decay stage.
    """
    
    def __init__(self,
                 lr=1e-2,
                 lr_warm=1e-3,
                 lr_end=1e-4,
                 warm_step=1e5,
                 increase_step=1e6,
                 decay_step=1e8):
        """Initialize
        
        Args:
            lr_warm (float):  The learning rate changes from 0 to lr_warm in the warm up stage.
            lr (float): The learning rate changes from lr_warm to lr in the increasing stage.
            lr_end (float): The learning rate changes from lr to lr_end in the decay stage.
            warm_step (int): The step between 0 and warm_step is in the warm up stage.
            increase_step (int): The step between warm_step and increase_step is in the increasing stage.
            decay_step (int): The step between warm_step and decay_step is in the decay stage.
        """
        super(LearningRate, self).__init__()

        self.lr = float(max(lr, 0.0))
        self.lr_warm = float(max(lr_warm, 0.0))
        self.lr_end = float(max(lr_end, 0.0))

        self.warm_step = float(max(warm_step, 0))
        self.increase_step = float(max(increase_step, 0))
        self.decay_step = float(max(decay_step, 0))

        self.step = 0

    def get_step(self):
        """Gets current training step.
        
        Returns:
            int: current training step.
        """
        return tf.to_float(tf.train.get_or_create_global_step())

    def _warm_up_lr(self, step):
        """Computes learning rate in the warm up stage.
        
        Args:
            step (int): current step.
        
        Returns:
            float: The updated learning rate.
        """
        return self.lr_warm * step / self.warm_step

    def _increase_lr(self, step):
        """Computes learning rate in the increasing stage.
        
        Args:
            step (int): current step.
        
        Returns:
            float: The updated learning rate.
        """
        ratio = (step - self.warm_step) / (self.increase_step - self.warm_step)
        return self.lr_warm + ratio * (self.lr - self.lr_warm)

    def _decay_lr(self, step):
        """Computes learning rate in the decay stage.
        
        Args:
            step (int): current step.
        
        Returns:
            float: The updated learning rate.
        """
        ratio = (step - self.increase_step) / \
            (self.decay_step - self.increase_step)
        return self.lr_end + (1.0 - ratio) * (self.lr - self.lr_end)

    def _end_lr(self, step):
        """Computes learning rate after the decay stage.
        
        Args:
            step (int): current step.
        
        Returns:
            float: The updated learning rate.
        """
        return self.lr_end

    def _less_than(self, a, b):
        """Returns the truth value of (a < b) element-wise.

        a is a Tensor, b is a float/int.
        
        Args:
            a (tensor): A tensor.
            b (float/int): A float or int value.`
        
        Returns:
            tensor: A tensor of type bool.
        """
        b = ops.convert_to_tensor(b, dtype=a.dtype.base_dtype)
        return tf.math.less(a, b)

    def get_lr(self):
        """Computes the learning rate according to the training step.
        
        Returns:
            float: The updated learning rate.
        """
        current_step = self.get_step()

        lr = tf.cond(
            self._less_than(current_step, self.warm_step),
            lambda: self._warm_up_lr(current_step),
            lambda: tf.cond(
                self._less_than(current_step, self.increase_step),
                lambda: self._increase_lr(current_step),
                lambda: tf.cond(
                    self._less_than(current_step, self.decay_step),
                    lambda: self._decay_lr(current_step),
                    lambda: self._end_lr(current_step)
                )
            )
        )

        return lr

    def __call__(self):
        return ops.convert_to_tensor(self.get_lr(), dtype=tf.float32)
