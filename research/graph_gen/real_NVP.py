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

"""RealNVP like graph generation procedure."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rev_GNN
import tensorflow as tf
import numpy as np
import copy

flags = tf.app.flags

FLAGS = flags.FLAGS

def mlp(inputs,
        layer_sizes,
        activation_fn=tf.nn.relu,
        output_act=None):
  prev_size = inputs.get_shape().as_list()[-1]
  shape_inp = tf.shape(inputs)
  if len(inputs.get_shape().as_list()) > 2:
    out = tf.reshape(inputs, [-1, shape_inp[2]])
  else:
    out = tf.reshape(inputs, [-1, shape_inp[1]])
  for i, layer_size in enumerate(layer_sizes):
    z = tf.layers.fully_connected(out, layer_size, activation_fn)

    if i < len(layer_sizes) - 1 and activation_fn is not None:
      out = activation_fn(z)
    elif i == len(layer_sizes) - 1 and output_act is not None:
      out = output_act(z)
    else:
      out = z
  if len(inputs.get_shape().as_list()) > 2:
    return tf.reshape(out, [shape_inp[0], shape_inp[1], -1])
  else:
    return tf.reshape(out, [shape_inp[0], -1])

def message_fn(params, model_hparams):
  """Messages are just MLPs of the nodes itself, as edges have no labels,
     so, message is just a transform of the node vector."""
  node_states = params['node_states']
  mask = params['mask']
  node_dim = model_hparams.node_dim
  msg_dim = model_hparams.msg_dim

  with tf.variable_scope('message_fn', reuse=tf.AUTO_REUSE):
    msg_function = mlp(node_states,
            layer_sizes=[model_hparams.msg_hidden1, model_hparams.msg_hidden2],
            activation_fn=tf.nn.tanh,
            output_act=tf.nn.tanh)
  # This could be a soft matrix, in which case we are taking a sum over all
  # possible edges/neighbours.
  adj_mat = params['adj_mat']
  temp_mask = tf.expand_dims(mask, 2)
  mask = tf.multiply(temp_mask, tf.transpose(temp_mask, (0, 2, 1)))
  adj_mat = adj_mat * mask

  if FLAGS.use_edge_features:
    edge_features = params['edge_feat']
    # Just to make sure that the edge features are interpreted in the way that
    # edge features give out a distribution over possible labels.
    if FLAGS.use_sigmoid_for_edge_feat:
      edge_features = tf.nn.sigmoid(edge_features)
    else:
      edge_features = tf.nn.softmax(edge_features)
    edge_features = edge_features * tf.expand_dims(mask, 3)
    edge_embd = params['edge_embd']
    # edge_features: B x N x N x (d+1)
    # edge_embd: (d+1) x m x m
    batch_size = tf.shape(edge_features)[0]
    num_nodes = tf.shape(edge_features)[1]
    print (edge_embd)
    edge_embd = tf.reshape(edge_embd,
                           [model_hparams.edge_dim + 1,
                            model_hparams.msg_dim*model_hparams.msg_dim])
    edge_features_rs = tf.reshape(edge_features,
                                  [batch_size*num_nodes*num_nodes,
                                   model_hparams.edge_dim + 1])
    edge_matrices = tf.matmul(edge_features_rs, edge_embd)
    edge_matrices = tf.reshape(edge_matrices,
                               [batch_size, num_nodes, num_nodes,
                                model_hparams.msg_dim, model_hparams.msg_dim])
    # edge_matrices:  B x N x N x d x d
    msg_tiled = tf.tile(tf.expand_dims(msg_function, 2),
                        [1, 1, tf.shape(msg_function)[1], 1])
    msg_edge_embd = tf.matmul(edge_matrices, tf.expand_dims(msg_tiled, 4))
    # B x N x N x d x 1
    msg_edge_embd = tf.multiply(tf.squeeze(msg_edge_embd, axis=4),
                                tf.expand_dims(adj_mat, 3))
    # B x N x N x d
    msg_function = tf.reduce_sum(msg_edge_embd, axis=1)
    return msg_function

  # adj_mat defacto handles the non-existance of particular nodes
  msg_function = tf.transpose(msg_function, perm=[0, 2, 1])
  msg_function = tf.matmul(msg_function, adj_mat)
  return tf.transpose(msg_function, perm=[0, 2, 1])

def aggn_fn(msg_matrix):
  """Messages are M(u -> v), so we need to sum over dimension 1."""
  return tf.reduce_sum(msg_matrix, 1)

def update_fn(params, model_hparams):
  node_states = params['node_states']
  msg_vec = params['agg_msg']
  mask = params['mask']
  batch_size = tf.shape(node_states)[0]
  num_nodes = tf.shape(node_states)[1]
  node_shape = tf.shape(node_states)
  msg_shape = tf.shape(msg_vec)

  is_training = params['is_training']

  msg_dim = model_hparams.msg_dim
  node_dim = model_hparams.node_dim
  print ('INFO: Msg Dim = ', msg_dim, ' Node Dim = ', node_dim)
  print ('Msg Vec = ', msg_vec)

  with tf.variable_scope('update_fn', reuse=tf.AUTO_REUSE):
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
    h_t = tf.multiply(1 - z_t, node_reshape) + tf.multiply(z_t, h_tilde)
    h_t_rs = tf.reshape(
        h_t, [batch_size, num_nodes, node_dim], name="h_t_rs")

    mask_col = tf.reshape(mask, [batch_size, num_nodes, 1])
    h_t_masked = tf.multiply(
        h_t_rs, mask_col, name="mul_h_t_masked"
    )

    h_t_rs = tf.reshape(
        h_t_masked, [batch_size, num_nodes, node_dim], name="h_t_rs_again")
    return h_t_rs

def message_passing_step(node_rep,
                         msg_fn, agg_fn, state_fn,
                         params, model_hparams):
  adj_mat = params['adj_mat']
  mask = params['mask']
  if not FLAGS.use_edge_features:
    params['edge_features'] = None
    params['edge_embd'] = None
  message_out = msg_fn(params={
                                'node_states' : node_rep,
                                'is_training' : params['is_training'],
                                'adj_mat'     : params['adj_mat'],
                                'edge_feat'   : params['edge_features'],
                                'edge_embd'   : params['edge_embd'],
                                'mask'        : params['mask']
                              },
                       model_hparams=model_hparams)
  agg_out = message_out
  next_state_output = update_fn(params={
                                         'node_states': node_rep,
                                         'is_training' : params['is_training'],
                                         'agg_msg' : agg_out,
                                         'mask': mask
                                        },
                                model_hparams=model_hparams)
  return next_state_output


class GraphNet(object):
  """Instantiation of Rev-GNN architecture for graph generation."""
  def __init__(self, hparams, params):
    self.hparams = hparams
    self.msg_fn = params['msg_fn'] if 'msg_fn' in params else message_fn
    self.agg_fn = params['agg_fn'] if 'agg_fn' in params else aggn_fn
    self.state_fn = params['update_fn'] if 'update_fn' in params else update_fn
    self.num_steps = self.hparams.num_steps
    self.is_training = params['is_training']
    self.n_edge_types = hparams.edge_dim
    self.n_node_types = hparams.node_dim

    if FLAGS.use_edge_features:
      with tf.variable_scope('edge_embd', reuse=tf.AUTO_REUSE):
        self.edge_embd = tf.get_variable(
            'edge_embd',
             shape=(self.n_edge_types+1,
                    hparams.msg_dim, hparams.msg_dim))

  def forward(self, node_states, adj_mat, mask, edge_feat=None):
    is_training = self.is_training
    def _f(node_rep):
      with tf.variable_scope('rev_mp/layer_0/b') as scope:
        params = dict()
        params['adj_mat'] = adj_mat
        params['is_training'] = is_training
        params['mask'] = mask
        if FLAGS.use_edge_features:
          params['edge_features'] = edge_feat
          params['edge_embd'] = self.edge_embd
        return message_passing_step(node_rep,
                                  self.msg_fn,
                                  self.agg_fn,
                                  self.state_fn,
                                  params,
                                  self.hparams)
    rev_mp = rev_GNN.rev_mp_block_backward(
                            node_states[:, :, :(self.hparams.node_dim)],
                            node_states[:, :, (self.hparams.node_dim):],
                            _f,
                            _f,
                            self.num_steps, is_training)
    node_states_out = tf.concat([rev_mp[0], rev_mp[1]], axis=2)
    return node_states_out

  def jacobian_forward(self, omega, adj_mat, mask=None, edge_feat=None):
    batch_size = tf.shape(omega)[0]
    return tf.zeros([batch_size], tf.float32)

  def jacobian_backward(self, omega, adj_mat, mask=None, edge_feat=None):
    batch_size = tf.shape(omega)[0]
    return tf.zeros([batch_size], tf.float32)

  def inverse(self, node_states, adj_mat, mask, edge_feat=None):
    is_training = self.is_training
    print ('Node states: ', node_states)
    def _f(node_rep):
      with tf.variable_scope('rev_mp/layer_0/b') as scope:
        params = dict()
        params['adj_mat'] = adj_mat
        params['is_training'] = is_training
        params['mask'] = mask
        if FLAGS.use_edge_features:
          params['edge_features'] = edge_feat
          params['edge_embd'] = self.edge_embd
        return message_passing_step(node_rep,
                                  self.msg_fn,
                                  self.agg_fn,
                                  self.state_fn,
                                  params,
                                  self.hparams)
    rev_mp = rev_GNN.rev_mp_block(
                            node_states[:, :, :(self.hparams.node_dim)],
                            node_states[:, :, (self.hparams.node_dim):],
                            _f,
                            _f,
                            self.num_steps, is_training)
    node_states_out = tf.concat([rev_mp[0], rev_mp[1]], axis=2)
    return node_states_out

class BatchNormBijector(tf.contrib.distributions.bijectors.BatchNormalization):
  """Extended batch norm bijector because of some changes that need to be
     made in the existing batch norm bijector implementation."""
  def __init__(self,
               batchnorm_layer=None,
               training=True,
               validate_args=False,
               name="batch_normalization"):
    print (name)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      g_constraint = lambda x: tf.nn.relu(x) + 1e-6
      batchnorm_layer = tf.layers.BatchNormalization(
          trainable=True,
          gamma_constraint=g_constraint,
          momentum=0.99,
          renorm_momentum=0.99,
          epsilon=1e-4)
      super(BatchNormBijector, self).__init__(batchnorm_layer,
                                              training,
                                              validate_args,
                                              name)

  def _forward(self, x):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      return self._de_normalize(x)

  def _normalize(self, y):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      return self.batchnorm.apply(y,
                                  training=(self._training and (not
                                                      FLAGS.batch_norm_type)))

  def _forward_log_det_jacobian(self, x):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      return -self._inverse_log_det_jacobian(x, use_saved_statistics=True)

  def _get_broadcast_fn(self, x):
    """Commenting out the sape fully defined condition, to ensure that
       full shape is not an issue."""
    # if not x.shape.is_fully_defined():
    #   raise ValueError("Input must have shape known at graph construction.")
    input_shape = tf.shape(x)
    input_shape_length = input_shape.get_shape().as_list()[0]

    ndims = input_shape_length
    reduction_axes = [i for i in range(ndims) if i not in self.batchnorm.axis]
    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.batchnorm.axis[0]] = (
        input_shape[self.batchnorm.axis[0]])
    def _broadcast(v):
      if (v is not None and
          len(v.get_shape()) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return tf.reshape(v, broadcast_shape)
      return v
    return _broadcast

  def _inverse_log_det_jacobian(self, y, use_saved_statistics=False):
    """Remove shape checking constraints, and make sure shape checking
       is lazily done at the time of running."""
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      input_shape = tf.shape(y)
      input_shape_length = input_shape.get_shape()[0]
      print ('Input shape length: ', input_shape_length)

      if not self.batchnorm.built:
        # Create variables.
        self.batchnorm.build(input_shape)

      event_dims = self.batchnorm.axis
      reduction_axes = [i for i in range(input_shape_length)\
                        if i not in event_dims]

      if (use_saved_statistics or not self._training) or FLAGS.batch_norm_type:
        log_variance = tf.log(
            self.batchnorm.moving_variance + self.batchnorm.epsilon)
      else:
        # At training-time, ildj is computed from the mean and log-variance
        # across the current minibatch.
        _, v = tf.nn.moments(y, axes=reduction_axes, keep_dims=True)
        log_variance = tf.log(v + self.batchnorm.epsilon)
      # `gamma` and `log Var(y)` reductions over event_dims.
      # Log(total change in area from gamma term).
      log_total_gamma = tf.reduce_sum(tf.log(self.batchnorm.gamma))

      # Log(total change in area from log-variance term).
      log_total_variance = tf.reduce_sum(log_variance)
      # The ildj is scalar, as it does not depend on the values of x and are
      # constant across minibatch elements.
      return log_total_gamma - 0.5 * log_total_variance

class BatchNormBijector2(object):
  """Bijector for applying Batch Normalisation.

  This class defines functions for running forward and backward passes over the
  batch norm bijector. Update ops were an issue with the regular tf batch norm
  bijector and hence custom BatchNormBijector created.

  Attributes:
    vars_created: Whether variables exist, or whether they should be created.
    scope: A tf.variable_scope with appropriate reuse settings to be used for
      defining ops and variables inside it.
    epsilon: A small floating point constant to prevent NaN from arising in
      computations when dividing by variance.
    name: Name of the instantiated object
    training: A bool whether instantiate in train mode or in test mode.
    decay: An float representing the decay factor in the moving mean and
      moving variance update in BatchNorm
  """

  def __init__(self,
               batchnorm_layer=None,
               training=True,
               validate_args=False,
               name='batch_normalization',
               mask=None):
    """Instantiate the batch normb ijector."""
    self._vars_created = False
    self._scope = tf.variable_scope(name, reuse=tf.AUTO_REUSE)
    self._epsilon = 1e-4
    self.name = name
    self._training = training
    self._decay = 0.42

  def _create_vars(self, x):
    """Create variables for this batch norm instance given a sample input."""
    n = tf.shape(x).get_shape().as_list()[0]
    n = x.get_shape().as_list()[n - 1]
    with self._scope:
      self.beta = tf.get_variable('beta', [1, n], trainable=True)
      self.gamma = tf.get_variable('gamma', [1, n], trainable=True)
      self.train_m = tf.get_variable(
          'moving_mean', [1, n],
          initializer=tf.zeros_initializer,
          trainable=False)
      self.train_v = tf.get_variable(
          'moving_var', [1, n],
          initializer=tf.ones_initializer,
          trainable=False)
    self._vars_created = True

  def _forward(self, x, **kwargs):
    """Run denormalisation (invert batch norm) for sampling computation."""
    if not self._vars_created:
      self._create_vars(x)
    x = tf.Print(
        x, [self.train_m, self.train_v],
        summarize=100,
        message='moving_mean_debug')
    return (x - self.beta) * tf.exp(-self.gamma) *\
                    tf.sqrt(self.train_v + self._epsilon) + self.train_m

  def _forward_log_det_jacobian(self, x, **kwargs):
    """Return log of the determinant of the jacobian equal to the negative of
       the inverse jacobian's log detetminant."""
    return -self._inverse_log_det_jacobian(x, **kwargs)

  def _inverse(self, x, use_saved_statistics=False, update_vars=False,
               **kwargs):
    """Apply BN in the forward direction during training."""
    if not self._vars_created:
      self._create_vars(x)
    _mask = kwargs['mask']
    input_shape = tf.shape(x)
    input_shape_length = input_shape.get_shape()[0]
    print('Input shape length: ', input_shape_length)

    reduction_axes = [ax_num for ax_num in range(input_shape_length - 1)]
    # At train time, use the current minibatch moments, at test time use the
    # moments from the moving average.
    if (use_saved_statistics or not self._training) or True:
      return (x - self.train_m) * 1. / tf.sqrt(self.train_v + self._epsilon) *\
                              tf.exp(self.gamma) + self.beta
    else:
      m, v = _masked_moments(
          x, axes=reduction_axes, mask=_mask, keep_dims=False)
      print('v in inverse: ', v)
      # Right now an exponential moving average is used. We could also
      # try something different.
      if update_vars:
        with self._scope:
          update_train_m = tf.assign_sub(
              self.train_m,
              self._decay * (self.train_m - m),
              use_locking=True,
              name='update_m')
          update_train_v = tf.assign_sub(
              self.train_v,
              self._decay * (self.train_v - v),
              use_locking=True,
              name='update_v')
        #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_train_v)
        #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_train_m)
        with tf.control_dependencies([update_train_m, update_train_v]):
          return (x - m)*1.0/tf.sqrt(v + self._epsilon)*tf.exp(self.gamma) +\
                                                                  self.beta
      else:
        return (x - m)*1.0/tf.sqrt(v + self._epsilon)*tf.exp(self.gamma) +\
                                                                  self.beta

  def _inverse_log_det_jacobian(self, x, **kwargs):
    """Compute the log determinant of the jacobian in the inverse pass."""
    input_shape = tf.shape(x)
    input_shape_length = input_shape.get_shape()[0]
    print('Input shape length: ', input_shape_length)
    _mask = kwargs['mask']

    reduction_axes = [ax_num for ax_num in range(input_shape_length - 1)]

    if not self._vars_created:
      self._create_vars(x)
    if self._training and False:
      _, v = _masked_moments(
          x, axes=reduction_axes, mask=_mask, keep_dims=False)
      v = tf.Print(v, [v], summarize=10, message='current-' + self.name)
      print('v: ', v)
    else:
      v = self.train_v
      v = tf.Print(
          v, [self.train_v], summarize=10, message='moving_' + self.name)
    log_det_jacobian = tf.reduce_sum(self.gamma) -\
                        0.5*tf.reduce_sum(tf.log(v + self._epsilon))
    return log_det_jacobian


class GraphCouplingBijector(object):
  """Coupling operation for graph based operations; where reversible GNNs are
     applied on top of models."""
  def __init__(self, gnn, adj_fn,
               event_ndims=0,
               validate_args=False,
               name="graph_coupling_bijector",
               hparams=None,
               params = None,
               is_training=True):
    assert hparams is not None
    assert params is not None
    self.graph_net = gnn if gnn is not None else gnn

    self.graph_net = GraphNet(hparams, params)
    self.adj_fn = adj_fn
    self.num_steps = hparams.num_steps
    self.name = name
    self.is_training = is_training

  def _forward(self, x, **kwargs):
    z, omega = x
    adj_mat = self.adj_fn(z)
    edge_feat = None
    if FLAGS.use_edge_features:
      assert 'edge_feat' in kwargs
      edge_feat = kwargs['edge_feat']
    # Takes as input the node features and adjacency matrix and does the
    # reversible message passing computation here.
    with tf.variable_scope("{name}".format(name=self.name), reuse=tf.AUTO_REUSE):
      omega_new = self.graph_net.forward(omega, adj_mat,
                                         kwargs['mask'], edge_feat)
    return (z, omega_new)

  def _forward_log_det_jacobian(self, x, **kwargs):
    z, omega = x
    mask = kwargs['mask']
    adj_mat = self.adj_fn(z)
    edge_feat = None
    if FLAGS.use_edge_features:
      edge_feat = kwargs['edge_feat']
    batch_size = tf.shape(z)[0]
    sum_log_det_jacobian = tf.zeros([batch_size])
    sum_log_det_jacobian += self.graph_net.jacobian_forward(omega, adj_mat,
                                                           mask=mask,
                                                           edge_feat=edge_feat)
    return sum_log_det_jacobian

  def _inverse(self, y, **kwargs):
    z, omega = y
    edge_feat = None
    if FLAGS.use_edge_features:
      edge_feat = kwargs['edge_feat']
    adj_mat = self.adj_fn(z)
    with tf.variable_scope("{name}".format(name=self.name), reuse=tf.AUTO_REUSE):
      omega_old = self.graph_net.inverse(omega, adj_mat,
                                         kwargs['mask'], edge_feat)
    return (z, omega_old)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    z, omega = y
    mask = kwargs['mask']
    edge_feat = None
    if FLAGS.use_edge_features:
      edge_feat = kwargs['edge_feat']
    batch_size = tf.shape(z)[0]
    sum_log_det_jacobian = tf.zeros([batch_size])
    sum_log_det_jacobian += self.graph_net.jacobian_backward(omega, z,
                                                            mask, edge_feat)
    return sum_log_det_jacobian

class CouplingBijector(object):
  def __init__(self, translation_fn, scale_fn,
               event_ndims=0,
               validate_args=False,
               name="coupling_bijector",
               is_training=True):
    self.translation_fn = translation_fn
    self.scale_fn = scale_fn
    self.name = name
    self.is_training = is_training
    self.n_edge_types = 5 ## Hardcoded for now
    if FLAGS.only_nodes:
      self.masking_translation = 0.0
    else:
      self.masking_translation = 1.0

  def _forward(self, x, **kwargs):
    z, omega = x
    mask = kwargs['mask']
    mask = tf.multiply(tf.expand_dims(mask, 2),
                       tf.transpose(tf.expand_dims(mask, 2), (0, 2, 1)))
    z_dims = tf.shape(z)[2]
    omega_new = omega

    with tf.variable_scope("{name}/scale".format(name=self.name),
                           reuse=tf.AUTO_REUSE):
      scale_omega = self.scale_fn(omega, z_dims)

    with tf.variable_scope("{name}/translation".format(name=self.name),
                           reuse=tf.AUTO_REUSE):
      translation_omega = self.translation_fn(omega, z_dims)
    exp_scale = tf.check_numerics(tf.exp(scale_omega*mask),
                                  " tf.exp(scale) is not numerically stable")
    if FLAGS.use_edge_features:
      z_update = translation_omega[0]
      edge_update = translation_omega[1]
    else:
      z_update = translation_omega
    z_new = (z*exp_scale +\
             self.masking_translation*z_update*mask)

    if FLAGS.use_edge_features:
      edge_feat = kwargs['edge_feat']
      temp_mask = tf.expand_dims(mask, 3)
      # like geometric mean of the edge features
      if FLAGS.use_scaling and FLAGS.share_scaling:
        edge_feat = FLAGS.lambda_combiner * edge_feat *\
          tf.expand_dims(exp_scale, 3) +\
          (1.0 - FLAGS.lambda_combiner)*edge_update * temp_mask
      else:
        edge_feat = FLAGS.lambda_combiner*edge_feat +\
                      (1.0 - FLAGS.lambda_combiner)*edge_update* temp_mask
      return (z_new, omega_new), edge_feat
    return (z_new, omega_new)

  def _forward_log_det_jacobian(self, x, **kwargs):
    z, omega = x
    mask = kwargs['mask']
    mask = tf.expand_dims(mask, 2)
    mask = tf.multiply(mask, tf.transpose(mask, (0, 2, 1)))

    z_dims = tf.shape(z)[2]
    with tf.variable_scope("{name}/scale".format(name=self.name),
                           reuse=tf.AUTO_REUSE):
      scale_omega = self.scale_fn(omega, z_dims)
    log_det_jacobian = tf.reduce_sum(scale_omega*mask, axis=[1,2])
    if FLAGS.use_edge_features:
      edge_feat = kwargs['edge_feat']
      if FLAGS.use_scaling and FLAGS.share_scaling:
        log_det_jacobian += FLAGS.lambda_combiner*(self.n_edge_types+1)*\
          tf.reduce_sum(scale_omega*mask, axis=tuple(range(1, len(z.shape))))
      else:
        log_det_jacobian += FLAGS.lambda_combiner*tf.reduce_sum(
            tf.ones_like(edge_feat)*tf.expand_dims(mask, 3), axis=[1, 2, 3])
    return log_det_jacobian

  def _inverse(self, y, **kwargs):
    z_new, omega_new = y
    mask = kwargs['mask']
    mask = tf.expand_dims(mask, 2)
    mask = tf.multiply(mask, tf.transpose(mask, (0, 2, 1)))
    z_dims = tf.shape(z_new)[2]
    omega = omega_new

    with tf.variable_scope("{name}/scale".format(name=self.name),
                           reuse=tf.AUTO_REUSE):
      scale_omega = self.scale_fn(omega, z_dims)

    with tf.variable_scope("{name}/translation".format(name=self.name),
                           reuse=tf.AUTO_REUSE):
      translation_omega = self.translation_fn(omega, z_dims)
    exp_scale = tf.check_numerics(tf.exp(-scale_omega*mask),
                                  " tf.exp(-scale) is not numerically stable")
    if FLAGS.use_edge_features:
      z_update = translation_omega[0]
      edge_update = translation_omega[1]
    else:
      z_update = translation_omega

    z = (z_new - self.masking_translation * z_update * mask)* exp_scale

    if FLAGS.use_edge_features:
      edge_feat = kwargs['edge_feat']
      temp_mask = tf.expand_dims(mask, 3)
      edge_feat = (edge_feat -\
                   (1.0 - FLAGS.lambda_combiner)*edge_update*temp_mask)
      if FLAGS.use_scaling and FLAGS.share_scaling:
        edge_feat = edge_feat*1.0/FLAGS.lambda_combiner*\
                          tf.expand_dims(exp_scale, 3)
      else:
        edge_feat = edge_feat*1.0/FLAGS.lambda_combiner
      return (z, omega), edge_feat

    return (z, omega)

  def _inverse_log_det_jacobian(self, y, **kwargs):
    z_new, omega_new = y
    mask = kwargs['mask']
    mask = tf.expand_dims(mask, 2)
    mask = tf.multiply(mask, tf.transpose(mask, (0, 2, 1)))
    z_dims = tf.shape(z_new)[2]
    with tf.variable_scope("{name}/scale".format(name=self.name),
                           reuse=tf.AUTO_REUSE):
      scale_omega = self.scale_fn(omega_new, z_dims)
    log_det_jacobian = -tf.reduce_sum(scale_omega*mask,
                                      axis=tuple(range(1, len(z_new.shape))))
    if FLAGS.use_edge_features:
      edge_feat = kwargs['edge_feat']
      if FLAGS.use_scaling and FLAGS.share_scaling:
        log_det_jacobian -= FLAGS.lambda_combiner*(self.n_edge_types+1)*\
          tf.reduce_sum(scale_omega*mask, axis=tuple(range(1, len(z.shape))))
      else:
        log_det_jacobian -= FLAGS.lambda_combiner*tf.reduce_sum(
            tf.ones_like(edge_feat), axis=[1, 2, 3])
    return log_det_jacobian


class RealNVP(object):

  def __init__(self,
               num_coupling_layers=2,
               event_ndims=0,
               name='real-nvp',
               variable_sharing=True):
    """Instantiate a realNVP bijector for graph generation."""
    self.num_coupling_layers = num_coupling_layers
    self.variable_sharing = variable_sharing
    self.name = name

  def build(self, params, hparams, adj_fn, translate_fn, scale_fn,
            is_training=None):
    num_coupling_layers = self.num_coupling_layers
    self.layers_z = [CouplingBijector(name="coupling_{i}_z".format(i=i),
                                      translation_fn=translate_fn,
                                      scale_fn=scale_fn,
                                      is_training=is_training)
                     for i in range(0, num_coupling_layers)]

    self.layers_omega = [GraphCouplingBijector(name="graph_{i}_h".format(i=i),
                                               gnn=None,
                                               adj_fn=adj_fn,
                                               hparams=hparams,
                                               params = params,
                                               is_training=is_training)
                     for i in range(0, num_coupling_layers)]

    # To use batch norm, we directly use batch norm bijector
    # We use two batch norm bijectors for the z and omega, because of the
    # different ways of applying batch norm to each of them
    self.layers_batch_norm_z = [
        BatchNormBijector(
            batchnorm_layer=None,
            training=is_training,
            validate_args=False,
            name='batch-norm-z-{i}'.format(i=i))
        for i in range(0, num_coupling_layers)]

    self.layers_batch_norm_h = [
        BatchNormBijector(
            batchnorm_layer=None,
            training=is_training,
            validate_args=False,
            name='batch-norm-h-{i}'.format(i=i))
        for i in range(0, num_coupling_layers)]

    self.layers_batch_norm_e = [
        BatchNormBijector(
            batchnorm_layer=None,
            training=is_training,
            validate_args=False,
            name='batch-norm-e-{i}'.format(i=i))
        for i in range(0, num_coupling_layers)]

    self.layers_h = self.layers_omega
    print ('RealNVP build finished....')

  def _forward(self, x, **kwargs):
    out = x
    # As we apply BN in the opposite direction, the first thing to do is to
    # invert Z using BN on the inputs, omega_0 is already normalised. Then move
    # ahead to get Z_0.5 which is expected to be normalised as omega_0 was. Now,
    # we apply the graph propagation onto omega_0.5 andZ_0.5 both of which are
    # normalised, so shouldn't shoot up by the end.
    for idx, (layer_z, layer_h) in enumerate(zip(self.layers_z,
                                                 self.layers_omega)):
      z, omega = out
      if not FLAGS.only_nodes:
        z = self.layers_batch_norm_z[idx]._forward(tf.expand_dims(z, 3))
        z  = tf.squeeze(z, 3)
        if FLAGS.use_edge_features:
          edge_feat = self.layers_batch_norm_e[idx]._forward(kwargs['edge_feat'])
      out = (z, omega)

      if not FLAGS.use_edge_features:
        out = layer_z._forward(out, **kwargs)
      else:
        kwargs['edge_feat'] = edge_feat
        out, edge_feat = layer_z._forward(out, **kwargs)
        kwargs['edge_feat'] = edge_feat

      z, omega = out
      if FLAGS.use_BN:
        omega = self.layers_batch_norm_h[idx]._forward(omega)
      out = (z, omega)

      out = layer_h._forward(out, **kwargs)

    if FLAGS.sample:
      return out, kwargs['edge_feat']
    return out

  def _forward_log_det_jacobian(self, x, **kwargs):
    z, omega = x
    sum_log_det_jacobian = 0
    out = x

    for idx, (layer_z, layer_h) in enumerate(zip(self.layers_z,
                                                 self.layers_omega)):
      # BN on Z
      if not FLAGS.only_nodes:
        z, omega = out
        sum_log_det_jacobian += self.layers_batch_norm_z[idx].forward_log_det_jacobian(
            tf.expand_dims(z, 3),
            event_ndims=1)
        z = self.layers_batch_norm_z[idx]._forward(tf.expand_dims(z, 3))
        z  = tf.squeeze(z, 3)
        if FLAGS.use_edge_features:
          sum_log_det_jacobian += self.layers_batch_norm_e[idx].forward_log_det_jacobian(
              kwargs['edge_feat'],
              event_ndims=1)
          edge_feat = self.layers_batch_norm_e[idx]._forward(kwargs['edge_feat'])
        out = (z, omega)

      # Z fprop
      sum_log_det_jacobian += layer_z._forward_log_det_jacobian(out, **kwargs)
      if FLAGS.use_edge_features:
        kwargs['edge_feat'] = edge_feat
        out, edge_feat = layer_z._forward(out, **kwargs)
        kwargs['edge_feat'] = edge_feat
      else:
        out = layer_z._forward(out, **kwargs)
      # BN on omega
      print ('KWARGS here: ', kwargs)
      z, omega = out
      if FLAGS.use_BN:
        sum_log_det_jacobian += self.layers_batch_norm_h[idx].forward_log_det_jacobian(
            omega,
            event_ndims=1)
        omega = self.layers_batch_norm_h[idx]._forward(omega)
      out = z, omega

      # omega fprop
      sum_log_det_jacobian += layer_h._forward_log_det_jacobian(out, **kwargs)
      out = layer_h._forward(out, **kwargs)

    return sum_log_det_jacobian

  def _inverse(self, y, **kwargs):
    z, omega = y
    self.layers_batch_norm_h.reverse()
    self.layers_batch_norm_z.reverse()
    self.layers_batch_norm_e.reverse()

    for idx, (layer_z, layer_h) in enumerate(zip(reversed(self.layers_z),
                                                 reversed(self.layers_h))):
      z, omega = layer_h._inverse((z, omega), **kwargs)
      if FLAGS.use_BN:
        omega = self.layers_batch_norm_h[idx]._inverse(omega)
      if FLAGS.use_edge_features:
        (z, omega), edge_feat = layer_z._inverse((z, omega), **kwargs)
        kwargs['edge_feat'] = edge_feat
      else:
        z, omega = layer_z._inverse((z, omega), **kwargs)
      if not FLAGS.only_nodes:
        if FLAGS.use_edge_features:
          edge_feat = self.layers_batch_norm_e[idx]._inverse(kwargs['edge_feat'])
          kwargs['edge_feat'] = edge_feat
        z = self.layers_batch_norm_z[idx]._inverse(tf.expand_dims(z, 3))
        z = tf.squeeze(z, 3)

    self.layers_batch_norm_h.reverse()
    self.layers_batch_norm_z.reverse()
    self.layers_batch_norm_e.reverse()
    if FLAGS.use_edge_features:
      return z, omega, edge_feat
    return z, omega

  def _inverse_log_det_jacobian(self, y, **kwargs):
    "Called during training -- Check twice for correctness."
    z, omega = y
    out = y
    sum_log_det_jacobian = 0
    self.layers_batch_norm_h.reverse()
    self.layers_batch_norm_z.reverse()
    self.layers_batch_norm_e.reverse()

    for idx, (layer_z, layer_h) in enumerate(zip(reversed(self.layers_z),
                                                 reversed(self.layers_h))):
      z, omega = out
      # omega_layer bprop
      sum_log_det_jacobian += layer_h._inverse_log_det_jacobian(out, **kwargs)
      out = layer_h._inverse(out, **kwargs)

      # omega-bn apply
      if FLAGS.use_BN:
        sum_log_det_jacobian += self.layers_batch_norm_h[idx].inverse_log_det_jacobian(
            omega,
            event_ndims=1)
        omega = self.layers_batch_norm_h[idx]._inverse(omega)
      out = (z, omega)
      # Z backprop
      sum_log_det_jacobian += layer_z._inverse_log_det_jacobian(out, **kwargs)
      if FLAGS.use_edge_features:
        out, edge_feat = layer_z._inverse(out, **kwargs)
        kwargs['edge_feat'] = edge_feat
      else:
        out = layer_z._inverse(out, **kwargs)

      # z-bn apply
      z, omega = out
      if not FLAGS.only_nodes:
        sum_log_det_jacobian += self.layers_batch_norm_z[idx].inverse_log_det_jacobian(
            tf.expand_dims(z, 3),
            event_ndims=1)
        z = self.layers_batch_norm_z[idx]._inverse(tf.expand_dims(z, 3))
        z = tf.squeeze(z, 3)
        if FLAGS.use_edge_features:
          sum_log_det_jacobian += self.layers_batch_norm_e[idx].inverse_log_det_jacobian(
              kwargs['edge_feat'],
              event_ndims=1)
          edge_feat = self.layers_batch_norm_e[idx]._inverse(kwargs['edge_feat'])
          kwargs['edge_feat'] = edge_feat
      out = (z, omega)

    self.layers_batch_norm_h.reverse()
    self.layers_batch_norm_z.reverse()
    self.layers_batch_norm_e.reverse()
    return sum_log_det_jacobian


def real_nvp_block(fn_params, num_layers):
  """Instantiate a Real NVP block for the graph generation process."""
  num_coupling_layers = params['num_coupling_layers']
  real_nvp = RealNVP(num_coupling_layers, name='real-nvp')
  return real_nvp

def real_nvp_model_fn(real_nvp_model, z,
                      omega, input_dist_fn, is_training=True, **kwargs):
  """Get the Real NVP density for corresponding arguments.
     Args:
       z: a batch of test sampled, adjacency matrix
       omega: a batch of test sampled, node features
  """
  print (kwargs)
  old_kwargs = copy.copy(kwargs)

  if FLAGS.use_edge_features:
    input_z, input_omega, edge_feat = real_nvp_model._inverse((z, omega),
                                                              **kwargs)
    kwargs['edge_feat'] = edge_feat
  else:
    input_z, input_omega = real_nvp_model._inverse((z, omega), **kwargs)
  log_prior_prob = input_dist_fn((input_z, input_omega), is_training, **kwargs)
  log_det_jacobian = real_nvp_model._inverse_log_det_jacobian((z,
                                                               omega),
                                                               **old_kwargs)
  log_posterior_prob = log_prior_prob + log_det_jacobian
  log_posterior_prob = tf.Print(log_posterior_prob, [log_prior_prob,
                                                     log_det_jacobian],
                                summarize=100,
                                message='Log Priors')

  return log_posterior_prob

def real_nvp_sample_fn(real_nvp_model, z_in, omega_in, input_dist_fn, **kwargs):
  """Start from the inputs and get the output sample."""
  print (kwargs)
  log_prior_prob = input_dist_fn((z_in, omega_in), **kwargs)
  out, edge_feat = real_nvp_model._forward((z_in, omega_in), **kwargs)
  log_det_jacobian = real_nvp_model._forward_log_det_jacobian((z_in, omega_in),
                                                              **kwargs)
  log_posterior_prob = log_prior_prob - log_det_jacobian
  log_posterior_prob = tf.Print(log_posterior_prob, [log_posterior_prob,
                                                     log_prior_prob,
                                                     log_det_jacobian],
                                summarize=100,
                                message='Log Priors')
  return log_posterior_prob, out, edge_feat


