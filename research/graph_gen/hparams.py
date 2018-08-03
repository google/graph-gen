# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Hparams for training the Rev GraphGen model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_hparams_ChEMBL():
  return tf.contrib.training.HParams(msg_dim=32,
                                     edge_features_present=False,
                                     opt_type='adam',
                                     max_grad_norm=4.0,
                                     num_upper_nvp=2,
                                     num_lower_nvp=2,
                                     use_dot_product_distance=False,
                                     use_similarity_in_space=False,
                                     omega_hidden1=10,
                                     omega_hidden2=10,
                                     msg_hidden1=32,
                                     msg_hidden2=32,
                                     combiner_hidden1=32,
                                     num_steps=2,
                                     omega_scale1=40,
                                     omega_scale2=20,
                                     weight_decay=0.001)

