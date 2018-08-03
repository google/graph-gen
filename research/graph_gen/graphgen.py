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
"""RealNVP generative model for molecule nets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import rev_GNN
import real_NVP

flags = tf.app.flags
flags.DEFINE_bool('trainable_variance', False,
                  'Whether to use trainable variance for the prior or not.')
flags.DEFINE_bool('use_discrete_dist', False,
                  'Whether to use discrete distribution for the prior.')
flags.DEFINE_bool('dirichlet', False,
                  'Whether to use Dirichlet prior or not for the omegas.')
flags.DEFINE_bool('beta_adj', False,
                  'Whether to use beta prior for the adjacency matrix.')
flags.DEFINE_bool('perturb_latent', False,
                  'Whether to make perturbations in the latent space.')
flags.DEFINE_bool('use_node_embedding', False,
                  'Whether to first convert nodes into a small embedding.')
flags.DEFINE_bool('time_dependent_prior', True,
                  'Whether to use prior which is time dependent.')
FLAGS = flags.FLAGS

class GraphGenerator(object):

  def __init__(self,
               hparams,
               params,
               name='graph-gen'):
    self.hparams = hparams
    self.node_dim = hparams.node_dim
    self.num_upper_nvp = hparams.num_upper_nvp
    self.num_lower_nvp = hparams.num_lower_nvp
    self.n_node_types = params['n_node_types']
    self.n_edge_types = params['n_edge_types']
    self.omega_alpha = 4.0
    self.omega_beta = 8.0
    self.z_alpha = 4.0
    self.z_beta = 8.0
    if FLAGS.use_edge_features:
      self.ef_beta = 4.0
      self.ef_alpha = 8.0
    # If we want to have a schedule over the prior shaprness, make it a new
    # variable.
    if not FLAGS.time_dependent_prior:
      return
    with tf.variable_scope('time_prior', reuse=tf.AUTO_REUSE):
      self.omega_alpha=tf.get_variable('omega_alpha', trainable=False,
                                      initializer=tf.constant(value=2.0))
      self.omega_beta = tf.get_variable('omega_beta', trainable=False,
                                       initializer=tf.constant(value=4.0))
      self.z_alpha = tf.get_variable('z_alpha', trainable=False,
                                    initializer=tf.constant(value=2.0))
      self.z_beta =tf.get_variable('z_beta', trainable=False,
                                  initializer=tf.constant(value=4.0))
      if FLAGS.use_edge_features:
        self.ef_alpha = tf.get_variable('ef_alpha', trainable=False,
                                      initializer=tf.constant(value=2.0))
        self.ef_beta = tf.get_variable('ef_beta', trainable=False,
                                     initializer=tf.constant(value=4.0))

  def assign_op(self, assign_ph):
    assgn_op1 = tf.assign(self.omega_alpha, assign_ph['omega_alpha'])
    assgn_op2 = tf.assign(self.omega_beta, assign_ph['omega_beta'])
    assgn_op3 = tf.assign(self.z_alpha, assign_ph['z_alpha'])
    assgn_op4 = tf.assign(self.z_beta, assign_ph['z_beta'])
    assgn_list = [assgn_op1, assgn_op2, assgn_op3, assgn_op4]
    if FLAGS.use_edge_features:
      assgn_op5 = tf.assign(self.ef_alpha, assign_ph['ef_alpha'])
      assgn_op6 = tf.assign(self.ef_beta, assign_ph['ef_beta'])
      assgn_list.append(assgn_op5)
      assgn_list.append(assgn_op6)
    return assgn_list

  def set_assign_placeholders(self):
    placeholders = dict()
    placeholders['omega_alpha'] = tf.placeholder(tf.float32)
    placeholders['omega_beta'] = tf.placeholder(tf.float32)
    placeholders['z_alpha'] = tf.placeholder(tf.float32)
    placeholders['z_beta'] = tf.placeholder(tf.float32)
    assgn_var_list = [self.omega_alpha, self.omega_beta,
                      self.z_alpha, self.z_beta]
    if FLAGS.use_edge_features:
      placeholders['ef_alpha'] = tf.placeholder(tf.float32)
      placeholders['ef_beta'] = tf.placeholder(tf.float32)
      assgn_var_list.append(self.ef_alpha)
      assgn_var_list.append(self.ef_beta)
    return placeholders, assgn_var_list

  def set_inputs(self):
    """Placeholders to be used while training the network."""
    self.z_in = tf.placeholder(tf.float32, [None, None, None])
    self.omega_in = tf.placeholder(tf.float32, [None, None, 2*self.node_dim])
    # Assumed to be just a number; inflate the matrix to use in case needed
    self.edge_features = tf.placeholder(tf.float32, [None, None,
                                                   None, self.n_edge_types+1])
    self.dropout_rate = tf.placeholder(tf.float32)
    self.mask = tf.placeholder(tf.float32, [None, None])
    placeholders = dict()
    placeholders['z_in'] = self.z_in
    placeholders['omega_in'] = self.omega_in
    placeholders['edge_in'] = self.edge_features
    placeholders['mask_in'] = self.mask
    return placeholders

  def setup(self, is_training=True, sample=False):
    real_nvp_estimator = real_NVP.RealNVP(
        num_coupling_layers=self.num_upper_nvp,
        event_ndims=0,
        name='real-nvp-estimator')
    params = dict()
    params['is_training'] = is_training
    if FLAGS.use_node_embedding:
      # First convert the node vectors into an embedding representation in
      # continuous space and then use it for the generation process. Note
      # that we need an invertible matrix transformation.
      omega_in = self.omega_in
      with tf.variable_scope('node_embedding', reuse=tf.AUTO_REUSE):
        shape = (2*self.node_dim, 2*self.node_dim)
        emb1 = tf.get_variable('emb1', shape=shape)
        emb2 = tf.get_variable('emb2', shape=shape)
        emb1 = emb1 * tf.convert_to_tensor(np.triu(np.ones(shape), k=1),
                                           dtype=tf.float32)
        emb2 = emb2 * tf.convert_to_tensor(np.tril(np.ones(shape), k=1),
                                           dtype=tf.float32)
        lower = tf.check_numerics(tf.exp(emb2),
                                  "tf.exp(emb2) is not numerically stable.")
        upper = tf.check_numerics(tf.exp(emb1),
                                  "tf.exp(emb1) is not numerically stable.")
        batch_size = tf.shape(omega_in)[0]
        num_nodes = tf.shape(omega_in)[1]
        temp_omega_in = tf.matmul(
            tf.reshape(self.omega_in, [-1, 2*self.node_dim]),
            lower)
        temp_omega_in = tf.matmul(temp_omega_in, upper)
        temp_omega_in = tf.reshape(temp_omega_in, [batch_size,
                                                   num_nodes,
                                                   2*self.node_dim])
        mask = tf.diag(tf.ones([2*self.node_dim]))
        log_det = tf.expand_dims(tf.reduce_sum(emb1*mask), 0)
        log_det += tf.expand_dims(tf.reduce_sum(emb2*mask), 0)
        self.log_det_inverse = log_det
        self.lower = lower
        self.upper = upper
        self.omega_val = temp_omega_in
    real_nvp_estimator.build(params, self.hparams,
                             adj_fn=self._adj_fn,
                             translate_fn=self._translation_fn,
                             scale_fn=self._scale_fn,
                             is_training=is_training)
    if sample:
      log_prob, out, edge_feat = self.sample_fn(real_nvp_estimator)
      return log_prob, out, edge_feat
    log_prob = self.model_fn(real_nvp_estimator, is_training)
    return log_prob

  def input_prior(self, x, is_training, **kwargs):
    z, omega = x
    if FLAGS.perturb_latent:
      z += tf.random_normal(tf.shape(z),
                          mean=0.0, stddev=0.05, dtype=tf.float32)
      omega += tf.random_normal(tf.shape(omega),
                                mean=0.0, stddev=0.05, dtype=tf.float32)
    batch_size = tf.shape(z)[0]
    num_nodes = tf.shape(omega)[1]
    mask_col = tf.reshape(self.mask, [batch_size, num_nodes, 1])
    mask_again = tf.multiply(mask_col,
                             tf.transpose(mask_col, (0, 2, 1)))

    # beta prior over transformed z
    # z -> log sigmoid(z)
    # Assuming sigmoid(z) is a sample from the beta prior, we get that
    # p(z) = p(sigmoid(z)) + log(sigmoid(z)) + log (1 - sigmoid(z))
    # where p(sigmoid(z)) is drawn from beta distribution.
    beta_dist = tf.distributions.Beta(concentration1=2.0,
                                          concentration0=2.0)
    log_sigmoid_z = -tf.nn.softplus(-z)
    unnorm_prob = ((beta_dist.concentration1 - 1.) * log_sigmoid_z
        + (beta_dist.concentration0 - 1.) * (-z + log_sigmoid_z))
    norm_const = (tf.lgamma(beta_dist.concentration1)
        + tf.lgamma(beta_dist.concentration0)
        - tf.lgamma(beta_dist.total_concentration))
    # beta prior value
    log_prob = unnorm_prob - norm_const
    # log (sigmoid(z)) term
    log_prob += log_sigmoid_z
    # log(1 - sigmoid(z)) term
    log_prob += (-z + log_sigmoid_z)
    log_prob = log_prob*mask_again
    log_density_z = tf.reduce_sum(log_prob, axis=[1,2])

    # We use the beta distribution prior on the input features z too,Log
    # where we try to enforce the fact that the model is unable to predict any
    # nodes initially, and only by virtue of the transformations it is able
    # to generate nodes and labels.
    beta_dist = tf.distributions.Beta(concentration1=2.0,
                                      concentration0=4.0)
    log_sigmoid_omega = -tf.nn.softplus(-omega)
    unnorm_prob = ((beta_dist.concentration1 - 1.) * log_sigmoid_omega
        + (beta_dist.concentration0 - 1.) * (-omega + log_sigmoid_omega))
    norm_const = (tf.lgamma(beta_dist.concentration1)
        + tf.lgamma(beta_dist.concentration0)
        - tf.lgamma(beta_dist.total_concentration))
    # beta prior value
    log_prob = unnorm_prob - norm_const
    # log (sigmoid(omega)) term
    log_prob += log_sigmoid_omega
    # log(1 - sigmoid(omega)) term
    log_prob += (-omega + log_sigmoid_omega)
    log_prob = log_prob*mask_col
    log_density_omega = tf.reduce_sum(log_prob, axis=[1,2])

    # We use the beta distribution for the edge features too, this means that
    # we draw sigmoid probabilities for each of the labels for the
    # edge features independently.
    log_density_edge = 0.0
    if FLAGS.use_edge_features:
      edge = kwargs['edge_feat']
      beta_dist = tf.distributions.Beta(concentration1=2.0,
                                      concentration0=2.0*(self.n_edge_types+1))
      log_sigmoid_edge = -tf.nn.softplus(-edge)
      unnorm_prob = ((beta_dist.concentration1 - 1.) * log_sigmoid_edge
          + (beta_dist.concentration0 - 1.) * (-edge + log_sigmoid_edge))
      norm_const = (tf.lgamma(beta_dist.concentration1)
          + tf.lgamma(beta_dist.concentration0)
          - tf.lgamma(beta_dist.total_concentration))
      # beta prior value
      log_prob = unnorm_prob - norm_const
      # log (sigmoid(omega)) term
      log_prob += log_sigmoid_edge
      # log(1 - sigmoid(omega)) term
      log_prob += (-edge + log_sigmoid_edge)
      log_prob = log_prob*tf.expand_dims(mask_again, 3)
      # log_prob = log_prob*tf.stop_gradient(tf.expand_dims(tf.nn.sigmoid(z),
      #                                                    axis=3))
      log_density_edge = tf.reduce_sum(log_prob, axis=[1,2, 3])

    total_log_prob = log_density_z + log_density_omega
    if FLAGS.use_edge_features:
      total_log_prob += log_density_edge
    total_log_prob = tf.Print(total_log_prob,
                              [tf.reduce_mean(total_log_prob),
                               tf.reduce_mean(log_density_z),
                               tf.reduce_mean(log_density_omega),
                               tf.reduce_mean(log_density_edge)],
                             message='prior_density', summarize=30)
    return total_log_prob

  def model_fn(self, real_nvp_estimator, is_training):
    omega_in = self.omega_in
    if FLAGS.use_node_embedding:
      omega_in = self.omega_val
    log_prob = real_NVP.real_nvp_model_fn(real_nvp_estimator,
                                          self.z_in,
                                          omega_in,
                                          self.input_prior,
                                          is_training,
                                          mask=self.mask,
                                          edge_feat=self.edge_features)
    if FLAGS.use_node_embedding:
      log_prob += self.log_det_inverse
    return log_prob

  def sample_fn(self, real_nvp_estimator):
    """Sample fn but needs to handle mask here too."""
    log_prob, out, edge_feat = real_NVP.real_nvp_sample_fn(real_nvp_estimator,
                                                  self.z_in,
                                                  self.omega_in,
                                                  self.input_prior,
                                                  is_training=False,
                                                  mask=self.mask,
                                                  edge_feat=self.edge_features)
    return log_prob, out, edge_feat

  def _translation_fn(self, omega, z_dims):
    """Now accounts for both an update in Z as well as an update in edge
       features."""
    if self.hparams.use_dot_product_distance:
      omega_t = tf.transpose(omega, perm=[0, 2, 1])
      similarity = tf.matmul(omega, omega_t)
      return similarity
    elif self.hparams.use_similarity_in_space:
      with tf.variable_scope('translation_fn', reuse=tf.AUTO_REUSE):
        h_omega = real_NVP.mlp(omega,
                       [self.hparams.omega_hidden1, self.hparams.omega_hidden2],
                       activation_fn=tf.nn.tanh,
                       output_nonlinearity=None,
                       regularizer=None)
        if FLAGS.l2_normalize:
          h_omega = tf.nn.l2_normalize(h_omega, dim=2)
        h_omega_t = tf.transpose(h_omega, perm=[0, 2, 1])
        similarity = tf.matmul(h_omega, h_omega_t)
        return similarity
    elif True:
      # Two headed neural net which gives me a score as well as a softmax
      # over edge features. We want to learn to generate Z using a combination
      # of the current node states as well as a context vector for all the
      # nodes.
      omega = tf.nn.sigmoid(omega)
      mask = tf.expand_dims(self.mask, 2)
      # Interpret omega as a distribution over labels.
      with tf.variable_scope('translation_fn', reuse=tf.AUTO_REUSE):
        h_omega = real_NVP.mlp(omega,
                       [self.hparams.omega_hidden1, self.hparams.omega_hidden2],
                       activation_fn=tf.nn.relu,
                       output_nonlinearity=None,
                       regularizer=None)

        with tf.variable_scope('context_scope', reuse=tf.AUTO_REUSE):
          context_omega = real_NVP.mlp(
              omega,
              [self.hparams.omega_hidden1, self.hparams.omega_hidden2],
              activation_fn=tf.nn.relu,
              output_nonlinearity=None,
              regularizer=None)
        context_omega = tf.reduce_sum(context_omega*mask, axis=1)
        h_omega_new = tf.expand_dims(h_omega*mask, axis=2)
        h_omega_perm = h_omega_new + tf.transpose(h_omega_new, perm=[0,2,1,3])
        batch_size = tf.shape(h_omega_perm)[0]
        num_nodes = tf.shape(h_omega_perm)[1]
        node_feat = tf.reshape(h_omega_perm, [batch_size,
                                              num_nodes*num_nodes,
                                              self.hparams.omega_hidden2])

        node_feat = tf.concat([node_feat,
                               tf.tile(tf.expand_dims(context_omega, 1),
                                       [1, num_nodes*num_nodes, 1])], axis=2)
        node_feat = tf.reshape(node_feat, [batch_size, num_nodes*num_nodes,
                                           2*self.hparams.omega_hidden2])
        with tf.variable_scope('node_features', reuse=tf.AUTO_REUSE):
          z_mat = real_NVP.mlp(node_feat,
                                 [self.hparams.combiner_hidden1, 1],
                                 activation_fn=tf.nn.tanh,
                                 output_nonlinearity=tf.log_sigmoid,
                                 regularizer=None)

        z_mat = tf.reshape(z_mat, [batch_size, num_nodes, num_nodes, 1])

        # Now the edge features, we know they come from n_node_dims
        if FLAGS.use_edge_features:
          with tf.variable_scope('edge_translation', reuse=tf.AUTO_REUSE):
            edge_feat = real_NVP.mlp(
                node_feat,
                [self.hparams.combiner_hidden1, self.n_edge_types+1],
                activation_fn=tf.nn.tanh,
                output_nonlinearity=tf.nn.log_softmax,
                regularizer=None)
          edge_feat = tf.reshape(edge_feat,
                                 [batch_size, num_nodes,
                                  num_nodes, self.n_edge_types+1])
          return tf.squeeze(z_mat, 3), edge_feat
        return tf.squeeze(z_mat, 3)

  def _scale_fn(self, omega, z_dims):
    return tf.zeros([tf.shape(omega)[0], tf.shape(omega)[1],
                     tf.shape(omega)[1]], dtype=tf.float32)
    with tf.variable_scope('scale_fn', reuse=tf.AUTO_REUSE):
      # omega = tf.reshape(omega, [tf.shape(omega)[0], -1])
      s_omega = real_NVP.mlp(omega,
                     [self.hparams.omega_scale1, self.hparams.omega_scale2],
                     activation_fn=tf.nn.tanh,
                     output_nonlinearity=None,
                     regularizer=None)
      s_omega_new = tf.expand_dims(s_omega, axis=2)
      s_omega_perm = s_omega_new + tf.transpose(s_omega_new, perm=[0, 2, 1, 3])
      batch_size = tf.shape(s_omega_perm)[0]
      num_nodes = tf.shape(s_omega_perm)[1]
      node_feat = tf.reshape(s_omega_perm, [batch_size, num_nodes*num_nodes,
                                            self.hparams.omega_scale2])

      with tf.variable_scope('final_scaling', reuse=tf.AUTO_REUSE):
        s_mat = real_NVP.mlp(
            node_feat,
            [self.hparams.combiner_hidden1, 1],
            activation_fn=tf.nn.tanh,
            output_nonlinearity=tf.log_sigmoid,
            regularizer=None)
      s_mat = tf.reshape(s_mat, [batch_size, num_nodes, num_nodes])
      return s_mat

  def _adj_fn(self, z_matrix):
    return tf.nn.sigmoid(z_matrix)

