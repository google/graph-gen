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

"""Reversible GNN architecture for Citation Network classification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import rev_GNN
import neumann_GNN as rbp
from adv_graphNN import mlp

flags = tf.app.flags
flags.DEFINE_bool('gresnet', False, 'Flag to specify whether to use'\
                  'one step forward residual connections.')
FLAGS = flags.FLAGS

def message_fn(params, model_hparams):
  """Convert Node features to messages along each edge.

  This function takes in the node features of each vertex and applies a function
  over them to generate the messages which are then aggregated over all
  incoming edges. There are no edge labels so messages are transforms of the
  node only.

  Args:
    params: A dict containing node_states, adj_mat, mask, edge embeddings and
      edge_features to be used in this function.
    model_hparams: A tf.HParams object containing the set of hparams for the
      model (like node_dim, msg_dim, etc).

  Returns:
    A [batch_size, num_nodes, msg_dim] sized tensor representing the aggregated
    messages coming at each node. The aggregation function is a simple addition.
  """
  node_states = params['node_states']
  node_dim = model_hparams.node_dim
  msg_dim = model_hparams.msg_dim
  with tf.variable_scope('message_fn', reuse=tf.AUTO_REUSE):
    msg_function = mlp(node_states, node_dim,
                       num_layers=2,
                       net_type='feed_forward',
                       activation_fn=tf.nn.relu,
                       output_units=[model_hparams.hidden1, msg_dim],
                       is_training=params['is_training'])
    msg_function = tf.nn.tanh(msg_function)
  msg_function = tf.transpose(msg_function, perm=[0, 2, 1])
  adj_mat = params['adj_mat']
  msg_function = tf.matmul(msg_function, adj_mat)
  return tf.transpose(msg_function, perm=[0, 2, 1])

def aggn_fn(msg_matrix):
  """Aggregate messages which are like M(u->v), so sum over dimension 1."""
  return tf.reduce_sum(msg_matrix, 1)

def update_fn(params, model_hparams):
  """Update the node states based on a GRU implementation.

  A standard GRU cell to update the hidden states of the nodes based on
  incoming aggregated messages from the graph.

  Args:
    params: A dict containing node_states, adj_mat, mask, edge embeddings and
      edge_features to be used in this function.
    model_hparams: A tf.HParams object containing the set of hparams for the
      model (like node_dim, msg_dim, etc).

  Returns:
    h_t_rs: Updates hidden states for each node, for each graph in the batch.
  """
  node_states = params['node_states']
  msg_vec = params['agg_msg']
  batch_size = tf.shape(node_states)[0]
  num_nodes = tf.shape(node_states)[1]
  node_shape = tf.shape(node_states)
  msg_shape = tf.shape(msg_vec)

  is_training = params['is_training']

  msg_dim = model_hparams.msg_dim
  node_dim = model_hparams.node_dim

  with tf.variable_scope('update_fn', reuse=tf.AUTO_REUSE):
    # Just in case we want to keep the update variables fixed.
    if FLAGS.mode == "diagnose":
      w_z = tf.get_variable("GRU_w_z", shape=[msg_dim, node_dim],
                            trainable=False)
      u_z = tf.get_variable("GRU_u_z", shape=[node_dim, node_dim],
                            trainable=False)
      w_r = tf.get_variable("GRU_w_r", shape=[msg_dim, node_dim],
                            trainable=False)
      u_r = tf.get_variable("GRU_u_r", shape=[node_dim, node_dim],
                            trainable=False)
      w = tf.get_variable("GRU_w", shape=[msg_dim, node_dim],
                          trainable=False)
      u = tf.get_variable("GRU_u", shape=[node_dim, node_dim],
                          trainable=False)
    else:
      w_z = tf.get_variable("GRU_w_z", shape=[msg_dim, node_dim])
      u_z = tf.get_variable("GRU_u_z", shape=[node_dim, node_dim])
      w_r = tf.get_variable("GRU_w_r", shape=[msg_dim, node_dim])
      u_r = tf.get_variable("GRU_u_r", shape=[node_dim, node_dim])
      w = tf.get_variable("GRU_w", shape=[msg_dim, node_dim])
      u = tf.get_variable("GRU_u", shape=[node_dim, node_dim])

    node_reshape = tf.reshape(node_states, [-1, node_shape[2]], name='node_rs')
    msg_reshape = tf.reshape(msg_vec, [-1, msg_shape[2]], name='msg_rs')

    z_t = tf.sigmoid(
        tf.matmul(msg_reshape, w_z) + tf.matmul(node_reshape, u_z), name="z_t")
    r_t = tf.sigmoid(
        tf.matmul(msg_reshape, w_r) + tf.matmul(node_reshape, u_r), name="r_t")

    h_tilde = tf.tanh(
        tf.matmul(msg_reshape, w) + tf.matmul(tf.multiply(r_t, node_reshape), u),
        name="h_tilde")

    # h_t has shape [batch_size * num_nodes, node_dim]
    h_t = tf.multiply(1 - z_t, node_reshape) + tf.multiply(z_t, h_tilde)
    h_t_rs = tf.reshape(
        h_t, [batch_size, num_nodes, node_dim], name="h_t_rs")

    return h_t_rs

def output_fn(params, model_hparams):
  """Generate the output using an output model from the GNN.

  Args:
    params: A dict containing node_states, adj_mat, mask, edge embeddings and
      edge_features to be used in this function.
    model_hparams: A tf.HParams object containing the set of hparams for the
      model (like node_dim, msg_dim, etc).

  Return:
    output_logits: The output logits for every node, class pair. These are not
      probabilities but just unnormalised log probs.
  """
  is_training = params['is_training']
  node_states = params['node_states']
  print ('Is-training: ', is_training)
  if FLAGS.batch_norm:
    with tf.variable_scope('batch_norm_scope', reuse=tf.AUTO_REUSE):
      node_states = tf.layers.batch_normalization(node_states,
                                                  center=True,
                                                  scale=True,
                                                  training=is_training)
  node_dim = model_hparams.node_dim
  num_classes = model_hparams.num_classes
  with tf.variable_scope('output_fn', reuse=tf.AUTO_REUSE):
    output_logits = mlp(node_states, node_dim,
                        num_layers=2,
                        net_type='feed_forward',
                        activation_fn=tf.nn.relu,
                        output_units=[2*num_classes, num_classes],
                        is_training=is_training)
  return output_logits

def message_passing_step(node_rep,
                         msg_fn, agg_fn, state_fn,
                         params, model_hparams):
  """Performs one step of message passing in the graph.

  Given the set of nodes in the graph, and pointers to functions for generating
  messages, aggregating messages, and update the states, runs one step of
  propagation.

  Args:
    node_rep: A tensor of size [batch_size, num_nodes, node_dim] ie node states
    msg_fn: A function which generates aggregated messages for the task
    agg_fn: A function which can aggregate messages given those along each edge
    state_fn: A function which can update the hidden states of the graph.
    params: A dict containing adj_mat, mask, edge embeddings and
      edge_features to be used in this function.
    model_hparams: A tf.HParams object containing the set of hparams for the
      model (like node_dim, msg_dim, etc).

  Returns:
    next_state_output: A tensor representing the states of the nodes after one
      step of propagation. If FLAGS.gresnet is set to True, then also add
      skip connection to the output value.
  """
  adj_mat = params['adj_mat']
  message_out = msg_fn(params={
                                'node_states' : node_rep,
                                'is_training' : params['is_training'],
                                'adj_mat'     : params['adj_mat']
                              },
                       model_hparams=model_hparams)
  # agg_out = agg_fn(message_out)
  agg_out = message_out
  next_state_output = update_fn(params={
                                         'node_states': node_rep,
                                         'is_training' : params['is_training'],
                                         'agg_msg' : agg_out,
                                        },
                                model_hparams=model_hparams)
  if FLAGS.gresnet:
    return next_state_output + node_rep
  else:
    return next_state_output


class RevMPNN(object):
  """Reversible MPNN model for the purpose of citation nets classification.

  Defines a reversible message passing Graph Net architecture to work on
  the problem of semi-supervised classification in graph nets.

  Attributes:
    hparams: A tf.HParams object containing the set of hyperparameters for use.
    old_hparams: A tf.Hparams object which is technically a copy of
      self.hparams to be used when the hparams are modified.
    msg_fn: A function which takes in node features and generates messages
      along each edge.
    agg_fn: A function which takes in messages along each edge and aggregates
      them and returns aggregated messages arriving at each node.
    state_fn: A function to update the node states using current state and
      aggregated messages.
    output_fn: A function to take in the states after propagation is done and
      then finally return a softmax over the classes for each node.
    num_steps: Number of steps of propagation.
    node_states: A tf.Placeholder which contains node states. Must be of shape:
      [batch_size, num_nodes, node_dim]
    adj_ph: A tf.Placeholder which contains adjacency matrix. Must be of shape:
      [batch_size, num_nodes, num_nodes]
  """
  def __init__(self, params, model_hparams):
    """Instantiate a reversible GNN for Citation networks."""
    self.hparams = model_hparams
    self.old_hparams = model_hparams
    self.msg_fn = params['msg_fn'] if 'msg_fn' in params else message_fn
    self.agg_fn = params['agg_fn'] if 'agg_fn' in params else aggn_fn
    self.state_fn = params['update_fn'] if 'update_fn' in params else update_fn
    self.output_fn = params['output_fn'] if 'output_fn' in params else output_fn
    self.num_steps = self.hparams.num_steps

  def set_inputs(self):
    """Set the input placeholders."""
    self.node_states = tf.placeholder(tf.float32,
                                      [None, None, self.hparams.node_dim],
                                      name = 'node_states')
    self.adj_ph = tf.placeholder(tf.float32, [None, None, None], name='adj_mat')

  def get_inputs(self):
    """Return the input placeholders as a dictionary."""
    retval = dict()
    retval['state'] = self.node_states
    retval['adjacency'] = self.adj_ph
    return retval

  def setup(self, is_training):
    """Set up the computation pipeline for the model."""
    # Use linear transform to learn lower dimensional feature projections.
    if FLAGS.use_linear_transform:
      with tf.variable_scope('init_transform', reuse=tf.AUTO_REUSE):
        _shape = tf.shape(self.node_states)
        print (self.old_hparams.node_dim)
        modified_nodes = tf.contrib.layers.fully_connected(
            tf.reshape(self.node_states, [-1, 1433]),
            self.hparams.operation_dim, None)
        modified_nodes = tf.reshape(modified_nodes, [_shape[0], _shape[1], -1])
        self.hparams.set_hparam('node_dim', self.old_hparams.operation_dim)
    if FLAGS.use_linear_transform:
      out = modified_nodes
    else:
      out = self.node_states
    def _f(node_rep):
      with tf.variable_scope("rev_mp/layer_0/b") as scope:
        params = dict()
        params['adj_mat'] = self.adj_ph
        params['is_training'] = is_training
        return message_passing_step(node_rep,
                                  self.msg_fn,
                                  self.agg_fn,
                                  self.state_fn,
                                  params,
                                  self.hparams)

    # Use the forward function of a reversible message passing block
    # to perform reversible message passing.
    rev_mp = rev_GNN.rev_mp_block(out,
                                  tf.zeros_like(out),
                                  _f,
                                  _f,
                                  self.num_steps, is_training)
    # Output function
    y0, y1 = rev_mp
    params = dict()
    params['is_training'] = is_training
    params['node_states'] = y1
    output_logits = self.output_fn(params, self.hparams)
    self.predicted_labels = tf.argmax(output_logits, 1)
    self.output = output_logits
    return {'logits': self.output,
            'labels': self.predicted_labels}

class VanillaMPNN(RevMPNN):
  """Regular GNNs for the purpose of document classification."""
  def __init__(self, params, model_hparams):
    """Instantiate a regular vanilla MPNN (with non-reverisble MP)."""
    RevMPNN.__init__(self, params, model_hparams)

  def setup(self, is_training):
    """Set up the computation pipeline for the model."""
    if FLAGS.use_linear_transform:
      with tf.variable_scope('init_transform', reuse=tf.AUTO_REUSE):
        _shape = tf.shape(self.node_states)
        print (self.old_hparams.node_dim)
        modified_nodes = tf.contrib.layers.fully_connected(
            tf.reshape(self.node_states, [-1, 1433]),
            self.hparams.operation_dim, None)
        modified_nodes = tf.reshape(modified_nodes, [_shape[0], _shape[1], -1])
        self.hparams.set_hparam('node_dim', self.old_hparams.operation_dim)
    if FLAGS.use_linear_transform:
      out = modified_nodes
    else:
      out = self.node_states
    params = dict()
    params['adj_mat'] = self.adj_ph
    params['is_training'] = is_training
    params['node_states'] = self.node_states


    # bag_of_nodes used if we want to directly train a feed forward
    # classifier over the inputs, without exploitting any relational
    # structure.
    if not FLAGS.bag_of_nodes:
      for i in range(2*self.num_steps):
        out = message_passing_step(out,
                                   self.msg_fn,
                                   self.agg_fn,
                                   self.state_fn,
                                   params,
                                   self.hparams)

    print (out, self.node_states)

    # Computing the jacobian spectrum
    # INSERT ANY JACOBIAN COMPUTING CODE HERE
    params = dict()
    params['is_training'] = is_training
    params['node_states'] = out
    output_logits  = self.output_fn(params, self.hparams)
    self.predicted_labels = tf.argmax(output_logits, 1)
    self.output = output_logits
    return {'logits': self.output,
            'labels': self.predicted_labels}

class NeumannMPNN(VanillaMPNN):
  """GNNs using Neumann RBP as the method for backpropagation."""

  def __init__(self, params, model_hparams):
    """Instantiate a regular neumann RBP based GNN."""
    VanillaMPNN.__init__(self, params, model_hparams)

  def setup(self, is_training):
    """Set up the computation piepline for the model."""
    out = self.node_states
    def _f(node_rep, is_trainable=True):
      with tf.variable_scope('neumann_mp/f', tf.AUTO_REUSE) as scope:
        params = dict()
        params['adj_mat'] = self.adj_ph
        params['is_training'] = is_training
        return message_passing_step(node_rep,
                                    self.msg_fn,
                                    self.agg_fn,
                                    self.state_fn,
                                    params,
                                    self.hparams)

    # Use forward pass for the Neumann RBP model, and this function call
    # defines the custom gradient for Neumann RBP as well.
    neumann_mp = rbp.neumann_mp_block(self.node_states,
                                      _f,
                                      max_num_steps=self.hparams.num_fprop_steps,
                                      rbp_steps=self.hparams.rbp_steps,
                                      is_training=is_training)

    params = dict()
    params['is_training'] = is_training
    params['node_states'] = neumann_mp
    output_logits  = self.output_fn(params, self.hparams)
    self.predicted_labels = tf.argmax(output_logits, 1)
    self.output = output_logits
    return {'logits': self.output,
            'labels': self.predicted_labels}
