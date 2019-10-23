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

"""Hparams for training the Rev GraphGen and Rev MPNN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_link_prediction_hparams():
  return tf.contrib.training.HParams(node_dim=60,
                                     msg_dim=60,
                                     max_grad_norm=4.0,
                                     num_steps=10,
                                     weight_decay=0.001,
                                     onehot_input=False,
                                     num_bases=2)

def get_ppi_hparams():
  return tf.contrib.training.HParams(learning_rate=5e-3,
                                     weight_decay=0.00004,
                                     node_dim=70,
                                     msg_dim=500,
                                     opt_type='adam',
                                     max_grad_norm=4.0,
                                     num_steps=2,
                                     num_classes=121,
                                     hidden1=80,
                                     rbp_steps=100,
                                     num_fprop_steps=100)

def get_citation_net_hparams():
  if FLAGS.dataset == 'cora':
    return tf.contrib.training.HParams(
        learning_rate=1e-4,
        weight_decay=0.001,
        node_dim=1433,
        msg_dim=800,
        operation_dim=2000,
        opt_type='adam',
        max_grad_norm=4.0,
        num_steps=2,
        hidden1=900,
        num_classes=7,
        rbp_steps=100,
        num_fprop_steps=100)
  elif FLAGS.dataset == 'pubmed':
    return tf.contrib.training.HParams(learning_rate=1e-4,
                                       weight_decay=0.001,
                                       node_dim=500,
                                       msg_dim=400,
                                       opt_type='adam',
                                       max_grad_norm=4.0,
                                       num_steps=2,
                                       hidden1=450,
                                       num_classes=3,
                                       rbp_steps=100,
                                       num_fprop_steps=100)
  else:
    return tf.contrib.training.HParams(learning_rate=1e-4,
                                       weight_decay=0.001,
                                       node_dim=3703,
                                       msg_dim=3000,
                                       opt_type='adam',
                                       max_grad_norm=4.0,
                                       num_steps=2,
                                       num_classes=6,
                                       hidden1=3200)

def get_convex_hull_hparams():
  return tf.contrib.training.HParams(node_dim=10,
                                     msg_dim=8,
                                     opt_type='adam',
                                     max_grad_norm=4.0,
                                     num_steps=50,
                                     hidden1=30,
                                     type='GRU',
                                     set_dependent=False)

def get_model_hparams():
  return tf.contrib.training.HParams(node_dim=50,
                                     msg_dim=100,
                                     edge_features_present=True,
                                     opt_type='adam',
                                     max_grad_norm=4.0,
                                     edge_num_layers=4, # TODO: check this value
                                     edge_hidden_dim=200,
                                     graph_output=True,
                                     num_steps=3,
                                     use_init_state_for_output=True,
                                     use_latest=True,
                                     out_hidden_dim=200,
                                     num_edge_types=4,
                                     non_edge=True,
                                     decay_start_point=0.4,
                                     final_learning_rate_factor=0.01)


def get_hparams(problem_name):
  if problem_name == "Convex_Hull":
    return get_convex_hull_hparams()
  elif problem_name == "CitationNet":
    return get_citation_net_hparams()
  elif problem_name == "Link-Prediction":
    return get_link_prediction_hparams()
  elif problem_name == "PPI":
    return get_ppi_hparams()
  else:
    return get_model_hparams()


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

