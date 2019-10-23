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

"""An advanced reversible GraphNN implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import rev_GNN
from hparams import get_model_hparams

flags = tf.app.flags
FLAGS = flags.FLAGS

HPARAMS = get_model_hparams()

def mlp(input_tensor, input_tensor_dim,
        num_layers=1,
        net_type='feed_forward',
        activation_fn=None,
        output_units=[HPARAMS.node_dim*HPARAMS.msg_dim],
        is_training=None):
  """Basic version of a MLP for use at many places."""
  input_tensor_shape = tf.shape(input_tensor)
  in_net = tf.reshape(input_tensor, [-1, input_tensor_dim], name='mlp1')
  out = in_net
  assert num_layers == len(output_units), "Mismatch in number of layers"

  for i in range(num_layers):
    with tf.name_scope('mlp_layer_%d' % (i)):
      if i == num_layers - 1:
        activation_fn = None
      out = tf.contrib.layers.fully_connected(out, output_units[i],
                                              activation_fn)

  out = tf.reshape(out,
                   [input_tensor_shape[0], input_tensor_shape[1], -1],
                   name='mlp3')
  return out

def message_fn(params):
  """Message generation from edge features, nodei representations."""
  keys = params.keys()
  adj_mat = params['adj_mat']
  if 'a_in' in keys and 'a_out' in keys:
    node_dim = HPARAMS.node_dim
    edge_dim = HPARAMS.msg_dim
    num_edge_types = HPARAMS.num_edge_types
    def _precompute_graph(adjacency_in):
      num_nodes = tf.shape(adjacency_in)[1]
      with tf.variable_scope("adjacency_mat", reuse=tf.AUTO_REUSE):
        matrices_in = tf.get_variable("adjacency_in",
                          shape=[num_edge_types, node_dim, node_dim])
        matrices_out = tf.get_variable("adjacency_out",
                          shape=[num_edge_types, node_dim, node_dim])

      zeros = tf.constant(0.0, shape=[1, node_dim, node_dim])
      if not HPARAMS.non_edge:
        matrices_in = tf.concat([zeros, matrices_in],
                                axis=0, name="matrices_in")
        matrices_out = tf.concat([zeros, matrices_out],
                                 axis=0, name="matrices_out")
      else:
        non_edge_in = tf.get_variable("non_edge_in",
                                      shape=[1, node_dim, node_dim])
        non_edge_out = tf.get_variable("non_edge_out",
                                       shape=[1, node_dim, node_dim])
        matrices_in = tf.concat([non_edge_in, matrices_in],
                                axis=0, name="matrices_in")
        matrices_out = tf.concat([non_edge_out, matrices_out],
                                 axis=0, name="matrices_out")
      adjacency_out = tf.transpose(adjacency_in, [0, 2, 1])
      a_in = tf.gather(matrices_in, tf.to_int32(adjacency_in))
      a_out = tf.gather(matrices_out, tf.to_int32(adjacency_out))

      a_in = tf.transpose(a_in, [0, 1, 3, 2, 4])
      a_in = tf.reshape(
        a_in, [-1, num_nodes * node_dim, num_nodes * node_dim])
      a_out = tf.transpose(a_out, [0, 1, 3, 2, 4])
      a_out = tf.reshape(
        a_out, [-1, num_nodes * node_dim, num_nodes * node_dim])
      return a_in, a_out

    with tf.variable_scope('message_fn', reuse=tf.AUTO_REUSE):
      batch_size = tf.shape(params['node_vec'])[0]
      num_nodes = tf.shape(params['node_vec'])[1]
      a_in, a_out = _precompute_graph(adj_mat)

      message_bias = tf.get_variable("message_bias", shape=HPARAMS.msg_dim)
      h_flat = tf.reshape(params['node_vec'],
                    [batch_size, num_nodes*HPARAMS.node_dim, 1], name="h_flat")
      a_in_mul = tf.reshape(
            tf.matmul(a_in, h_flat), [batch_size * num_nodes, HPARAMS.node_dim],
            name="a_in_mul")
      a_out_mul = tf.reshape(
            tf.matmul(a_out, h_flat), [batch_size * num_nodes,HPARAMS.node_dim],
            name="a_out_mul")
      a_temp = tf.concat(
            [a_in_mul, a_out_mul], axis=1,
            name="a_temp")  # shape [batch_size * num_nodes, 2*node_dim]
      a_t = a_temp + message_bias
      messages = tf.reshape(a_t, [batch_size, num_nodes, 2*HPARAMS.node_dim])
    return messages

  elif 'edge_features' in keys and params['edge_features'] is not None:
    print ('Comes here')
    with tf.variable_scope('message_fn', reuse=tf.AUTO_REUSE):
      # Based on MPNNs: M(h_u, h_v, e_uv) = A(e_uv) h_u
      edge_placeholder = params['edge_features']
      edge_shape = tf.shape(edge_placeholder)

      output_units = [HPARAMS.edge_hidden_dim]*(
                HPARAMS.edge_num_layers- 1) + [HPARAMS.node_dim*HPARAMS.msg_dim]
      A_euv = mlp(edge_placeholder, edge_placeholder.get_shape().as_list()[3],
                  num_layers=HPARAMS.edge_num_layers,
                  net_type='feed_forward',
                  activation_fn=tf.nn.tanh,
                  is_training=params['is_training'],
                  output_units=output_units)
      A_euv_s = tf.shape(A_euv)
      A_euv_new = tf.reshape(A_euv,
        [A_euv_s[0], A_euv_s[1], A_euv_s[2], HPARAMS.msg_dim, HPARAMS.node_dim])
      # A_euv_new: B x N x N x d_msg x d_node

      h_new = tf.expand_dims(params['node_vec'], 3)
      h_new = tf.expand_dims(h_new, 2)
      # h_new: B x N x 1 x d x 1
      h_new = tf.tile(h_new, [1, 1, tf.shape(h_new)[1], 1, 1], name='h_tile')

      output = tf.matmul(A_euv_new, h_new, name='msg_matmul')
      # output: B x N x N x d_msg x 1
      output = tf.squeeze(output, 4)
    return output

  else:
    with tf.variable_scope('message_fn', reuse=tf.AUTO_REUSE):
      # M (h_u, h_v) = MLP(h_u)
      node_placeholder = params['node_vec']
      node_shape = tf.shape(node_placeholder)

      node_mlp = mlp(node_placeholder, node_shape[2],
                     num_layers=1,
                     net_type='feed_forward',
                     activation=tf.nn.tanh,
                     is_training=params['is_training'])

      # node_msg: B x N x d_msg
      output = tf.tile(tf.expand_dims(node_mlp, 2), [1, 1, node_shape[1], 1])
    return output

def aggregation_fn(msg_matrix):
  """Aggregate the messages inputs sent across edges."""
  return msg_matrix
  return tf.reduce_sum(msg_matrix, 1)

def update_fn(params):
  """A general update rule for states (GRU like update)."""
  assert 'node_vec' in params, "Node vectors not found"
  assert 'agg_msg' in params, "Aggregated message not found"
  assert 'mask' in params, "No way to determine which nodes exist"

  node_vec = params['node_vec']
  batch_size = tf.shape(node_vec)[0]
  num_nodes = tf.shape(node_vec)[1]
  msg_vec = params['agg_msg']
  # msg_vec is of the dimension:  B x N x d_msg

  node_shape = tf.shape(node_vec)
  msg_shape = tf.shape(msg_vec)
  mask = params['mask']
  mask_col = tf.cast(
      tf.reshape(mask, [-1, 1]), tf.float32, name='mask_col')

  is_training = params['is_training']

  msg_dim = HPARAMS.msg_dim
  node_dim = HPARAMS.node_dim
  print ('INFO: Msg Dim = ', msg_dim, ' Node Dim = ', node_dim)

  with tf.variable_scope('update_fn', reuse=tf.AUTO_REUSE):
    w_z = tf.get_variable("GRU_w_z", shape=[msg_dim, node_dim])
    u_z = tf.get_variable("GRU_u_z", shape=[node_dim, node_dim])
    w_r = tf.get_variable("GRU_w_r", shape=[msg_dim, node_dim])
    u_r = tf.get_variable("GRU_u_r", shape=[node_dim, node_dim])
    w = tf.get_variable("GRU_w", shape=[msg_dim, node_dim])
    u = tf.get_variable("GRU_u", shape=[node_dim, node_dim])

    node_reshape = tf.reshape(node_vec, [-1, node_shape[2]], name='node_rs')
    msg_reshape = tf.reshape(msg_vec, [-1, msg_shape[2]], name='msg_rs')

    z_t = tf.sigmoid(
        tf.matmul(msg_reshape, w_z) + tf.matmul(node_reshape, u_z), name="z_t")
    r_t = tf.sigmoid(
        tf.matmul(msg_reshape, w_r) + tf.matmul(node_reshape, u_r), name="r_t")

    h_tilde = tf.tanh(
        tf.matmul(msg_reshape, w) + tf.matmul(tf.multiply(r_t, node_reshape), u),
        name="h_tilde")

    h_t = tf.multiply(1 - z_t, node_reshape) + tf.multiply(z_t, h_tilde)
    h_t_masked = tf.multiply(
        h_t, mask_col, name="mul_h_t_masked"
    )
    h_t_rs = tf.reshape(
        h_t_masked, [batch_size, num_nodes, HPARAMS.node_dim], name="h_t_rs")
    return h_t_rs

def rev_output_fn(params):
  """Output function for combining node features to graph level output."""
  is_training = params['is_training']
  node_states = params['node_states']
  mask = params['mask']
  mask = tf.expand_dims(mask, axis=2)
  graph_output_size = params['graph_output_size']
  with tf.variable_scope('rev_output_fn', reuse=tf.AUTO_REUSE):
    # Based on the Paper of GGNN (Gated Graph Neural Networks), Li. et.al.
    with tf.variable_scope('i_function', reuse=tf.AUTO_REUSE):
      i_output = mlp(node_states, node_states.get_shape().as_list()[2],
                     num_layers=2,
                     activation_fn=tf.nn.relu,
                     output_units=[HPARAMS.out_hidden_dim, graph_output_size],
                     is_training=is_training)

    with tf.variable_scope('j_function', reuse=tf.AUTO_REUSE):
      j_output = mlp(node_states, node_states.get_shape().as_list()[2],
                     num_layers=2,
                     activation_fn=tf.nn.relu,
                     output_units=[HPARAMS.out_hidden_dim, graph_output_size],
                     is_training=is_training)

      gated_activations = tf.multiply(tf.nn.sigmoid(i_output), j_output)

    gated_activations = tf.multiply(gated_activations, tf.to_float(mask))
    batch_size = tf.shape(node_states)[0]
    num_nodes = tf.shape(node_states)[1]

    gated_activations = tf.reshape(gated_activations,
                                   [batch_size, num_nodes, -1])

    output = tf.reduce_sum(gated_activations, axis=1)
    return output

def message_passing_step(node_rep, msg_fn, agg_fn, state_fn, params):
  """Define one step of message passing in the general GNN."""
  adj_mat = params['adj_mat']
  edge_features = params['edge_features'] if 'edge_features' in params else None
  message_out = msg_fn(params={
                                'adj_mat': params['adj_mat'],
                                'edge_features': edge_features,
                                'node_vec': node_rep,
                                'is_training': params['is_training'],
                                'a_in': params['a_in'],
                                'a_out': params['a_out']
                              })

  aggregated_output = agg_fn(message_out)
  mask = params['mask']
  next_state_output = update_fn(params={
                                        'node_vec': node_rep,
                                        'agg_msg': aggregated_output,
                                        'mask': mask,
                                        'is_training': params['is_training']
                                })
  if FLAGS.gresnet:
    return next_state_output + node_rep
  return next_state_output


class RevMPNN(object):
  """Class to define and run a MPNN (GGNN) which is reversible."""
  def __init__(self, params):
    """Instantiates a reversible version of MPNN."""
    # Functions for graph updates
    self.msg_fn = params['msg_fn']
    self.agg_fn = params['agg_fn']
    self.state_fn = params['state_fn']
    self.output_fn = params['output_fn']

    # Hyperparameters
    self.num_steps = params['num_steps']
    self.node_dim = params['node_dim']
    self.msg_dim = params['msg_dim']
    self.input_node_dim = params['input_node_dim']
    self.graph_output_size = params['output_dim']
    self.edge_dim = params['edge_dim']

    # Whether or not setting edge_features
    self.edge_features_present = params['edge_features_present']
    self.edge_features_in = None
    if self.edge_features_present:
      self.edge_dim = params['edge_dim']

  def _pad_fn(self, node_input_states):
    """Pad zero dimensions to the inputs for the purpose of expressivity."""
    input_node_dim = node_input_states.get_shape().as_list()[2]
    if input_node_dim > self.node_dim and not FLAGS.reduce_dim:
      raise ValueError("input_node_dim (%d) must be <= hparams.node_dim (%d)" %
                       (input_node_dim, self.node_dim))
    padded_nodes = tf.pad(node_input_states, [
        [0, 0],
        [0, 0],
        [0, max(self.node_dim - input_node_dim, 0)],
    ])
    return padded_nodes

  def set_inputs(self):
    """Set input placeholders for the reversible GNN model."""
    self.node_input_states = tf.placeholder(tf.float32,
                                      [None, None, self.input_node_dim],
                                      name="node_states")
    self.node_states = self._pad_fn(self.node_input_states)
    self.mask = tf.placeholder(tf.bool, shape=[None, None], name='mask')
    self.adj_ph = tf.placeholder(tf.float32, [None, None, None], name='adj_mat')
    if self.edge_features_present:
      self.edge_features_in = tf.placeholder(tf.float32,
                                             [None, None, None, self.edge_dim])

  def get_inputs(self):
    """Return a dictionary which maps input names to placeholders."""
    retval = dict()
    retval['adjacency'] = self.adj_ph
    retval['state'] = self.node_input_states
    retval['mask'] = self.mask
    if self.edge_features_present:
      retval['edge_features_in'] = self.edge_features_in
    return retval

  def _precompute_graph(self, adjacency_in, matrices_in, matrices_out):
    """Code for preprocessing adjacency matrix."""
    num_nodes = tf.shape(adjacency_in)[1]
    print ('Number of edge labels: ', self.edge_dim)
    #with tf.variable_scope("adjacency_mat", reuse=tf.AUTO_REUSE):
    # matrices_in = tf.get_variable("adjacency_in",
    #      shape=[self.edge_dim, self.node_dim, self.node_dim])

    # matrices_out = tf.get_variable("adjacency_out",
    #       shape=[self.edge_dim, self.node_dim, self.node_dim])

    zeros = tf.constant(0.0, shape=[1, self.node_dim, self.node_dim])
    matrices_in = tf.concat([zeros, matrices_in], axis=0, name="matrices_in")

    matrices_out = tf.concat([zeros, matrices_out], axis=0, name="matrices_out")

    adjacency_out = tf.transpose(adjacency_in, [0, 2, 1])
    a_in = tf.gather(matrices_in, tf.to_int32(adjacency_in))
    a_out = tf.gather(matrices_out, tf.to_int32(adjacency_out))

    # Make a_in and a_out have shape [batch_size, n*d, n*d]
    # the node repsentations are shape [batch_size, n*d] so we can use
    # tf.matmul with a_in and the node vector
    # The transpose is necessary to make the reshape read the elements of A
    # in the correct order (reshape reads lexicographically starting with
    # index [0][0][0][0] -> [0][0][0][1] -> ...)
    a_in = tf.transpose(a_in, [0, 1, 3, 2, 4])
    self._a_in = tf.reshape(
        a_in, [-1, num_nodes * self.node_dim, num_nodes * self.node_dim])
    a_out = tf.transpose(a_out, [0, 1, 3, 2, 4])
    self._a_out = tf.reshape(
        a_out, [-1, num_nodes * self.node_dim, num_nodes * self.node_dim])

  def setup(self, is_training, *args):
    """Set up the computation graph for the model,"""
    def _f(node_rep):
      with tf.variable_scope("rev_mp/layer_0/b") as scope:
        #self._precompute_graph(self.adj_ph)
        params = dict()
        params['adj_mat'] = self.adj_ph
        params['edge_features'] = (self.edge_features_in if
                            self.edge_features_in is not None else None)
        params['mask'] = self.mask
        params['is_training'] = is_training
        # params['a_in'] = self._a_in
        # params['a_out'] = self._a_out
        # with tf.variable_scope("adj_compute"):
        #   matrices_in = tf.get_variable("adjacency_in",
        #       shape=[self.edge_dim, self.node_dim, self.node_dim])
        #  matrices_out = tf.get_variable("adjacency_out",
        #       shape=[self.edge_dim, self.node_dim, self.node_dim])
        #  self._precompute_graph(self.adj_ph, matrices_in, matrices_out)
        params['a_in'] = None
        params['a_out'] = None
        return message_passing_step(node_rep,
                                    self.msg_fn,
                                    self.agg_fn,
                                    self.state_fn,
                                    params)

    rev_mp = rev_GNN.rev_mp_block(self.node_states,
                                  self.node_states,
                                  _f,
                                  _f,
                                  self.num_steps, is_training)

    y0, y1 = rev_mp
    params = dict()
    params['mask'] = self.mask
    params['is_training'] = is_training
    if HPARAMS.use_init_state_for_output:
      if HPARAMS.use_latest:
        combined_node_rep = tf.concat([y1, self.node_states], axis=2)
      else:
        temp0 = tf.expand_dims(y0, axis=3)
        temp1 = tf.expand_dims(y1, axis=3)
        max_pool = tf.reduce_max(tf.concat([temp0, temp1], axis=3), axis=3)
        combined_node_rep = tf.concat([max_pool, self.node_states], axis=2)
    else:
      combined_node_rep = tf.concat([y0, y1], axis=2)
    params['node_states'] = combined_node_rep
    params['graph_output_size'] = self.graph_output_size
    self.output_rep = self.output_fn(params)

    if HPARAMS.graph_output:
      self.output_final = self.output_rep
    return self.output_final

class RegularMPNN(RevMPNN):
  """Class to run regular MPNN model on the QM9 dataset."""
  def __init__(self, params):
    RevMPNN.__init__(self, params)

  def setup(self, is_training, *args):
    '''Setup function for regular message passing.'''
    out = self.node_states
    params = dict()
    params['adj_mat'] = self.adj_ph
    params['edge_features'] = (self.edge_features_in if
                            self.edge_features_in is not None else None)
    params['mask'] = self.mask
    params['is_training'] = is_training
    params['a_in'] = None
    params['a_out'] = None
    for i in range(2*self.num_steps):
      out = message_passing_step(out,
                                 self.msg_fn,
                                 self.agg_fn,
                                 self.state_fn,
                                 params)

    if HPARAMS.use_init_state_for_output:
      combined_node_rep = tf.concat([out, self.node_states], axis=2)
    else:
      assert False, "No other option available"
    params = dict()
    params['mask'] = self.mask
    params['is_training'] = is_training
    params['node_states'] = combined_node_rep
    params['graph_output_size'] = self.graph_output_size
    self.output_rep = self.output_fn(params)

    if HPARAMS.graph_output:
      self.output_final = self.output_rep
    return self.output_final

