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

"""Training script for realNVP generative model for graphs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn import metrics
import os
import graphgen
import hparams
import data_utils as mdu

flags = tf.app.flags
flags.DEFINE_integer('seed', 8, 'Random seed')
flags.DEFINE_integer('train_steps', 200, 'Maximum number of train iterations.')
flags.DEFINE_float('learning_rate', 1e-5, 'Learning rate for the optimizer.')

flags.DEFINE_integer('batch_size', 20, 'Batch size for training')
flags.DEFINE_integer('eval_every', 100,
                     'Number of epochs after which to evaluate')
flags.DEFINE_string('exp_dir', './citation_net_rev_mpnn/',
                    'Directory where to store results and checkpoints')
flags.DEFINE_bool('clip_gradients', False, 'Whether to clip gradients or not')
flags.DEFINE_string('data_dir', '/cns/vz-d/home/aviralkumar/citation_data',
                    'Data directory for the dataset')
flags.DEFINE_string('dataset', 'max20', 'Dataset to tesst the model on.')
flags.DEFINE_bool('add_weight_decay', True, 'Whether to add weight decay')
flags.DEFINE_integer('n_parallel_paths', 1, 'Number of parallel paths for'
                      'in-graph parallelization')
flags.DEFINE_bool('l2_normalize', False, 'Whether to perform a L2-normalization'
                  'before computing dot product to update Z')
flags.DEFINE_bool('only_nodes', False, 'Whether to use only nodes and not'
                  'the adjacency matrix features in the model.')
flags.DEFINE_bool('use_BN', False, 'Whether to use BN or not for nodes')
flags.DEFINE_bool('perturb_data', False,
                  'Whether to perturb data before passing it through.')
flags.DEFINE_bool('use_edge_features', False,
                  'Whether to use edge features or not for generation.')
flags.DEFINE_bool('symmetric', True,
                  'Whether the graph is symmetric (undirected) or directed.')
flags.DEFINE_float('lambda_combiner', 0.5,
                   'Combiner for taking the geometric mean of edge features.')
flags.DEFINE_integer('num_samples', 100,
                     'Number of samples to take out during prediction.')
flags.DEFINE_bool('sample', False,
                  'Whether to run in the inference (sample) mode or in train.')
flags.DEFINE_bool('jacobian_penalty', False,
                  'Whether to apply jacobian penalty or not.')
flags.DEFINE_bool('use_sigmoid_for_edge_feat', False,
                  'Whether to use sigmoid for edge features or softmax')
flags.DEFINE_bool('use_scaling', True,
                  'Whether use scaling function in edge_feat or adj update.')
flags.DEFINE_bool('share_scaling', False,
                  'Whether to share scaling between adj_mat and edge_feat.')
flags.DEFINE_integer('sharpen_steps', 10,
                     'Interval in which to sharpen the alphas and betas.')
flags.DEFINE_bool('batch_norm_type', True,
                  'Whether to have ghost batch_norm or not.')
flags.DEFINE_bool('only_score', False,
                  'Whether to just run the scoring part with no training.')
flags.DEFINE_bool('measure_ROC', False,
                  'Whether to measure the ROC curve for the model.')
FLAGS = flags.FLAGS

def _print_trainable_vars():
  print ('----------------------------------')
  print ('Trainable/All Variables: ')
  for x in tf.trainable_variables():
    print (x)
  print ('----------------------------------')

def add_to_feed_dict(placeholder, value, feed_dict):
  if isinstance(placeholder, tf.Tensor):
    feed_dict[placeholder] = value
  else:
    for p, v in zip(placeholder, value):
      add_to_feed_dict(p, v, feed_dict)

def fill_feed_dict(*pairs):
  feed_dict = {}
  for placeholder, value in pairs:
    add_to_feed_dict(placeholder, value, feed_dict)
  return feed_dict

def _print_matrix_function(matrix):
  for i in matrix:
    print(i.tolist())

def get_data_for_pass(dataset, placeholders, permute=False,
                      n_node_types=None,
                      n_edge_types=None, fake_data=False):
  assert n_node_types is not None
  assert n_edge_types is not None
  batch_molecules = dataset.next()
  adj_batch = []
  omega_batch = []

  n_nodes_batch = []
  max_graph_size = max(np.array([len(m.atoms) for m in batch_molecules]))
  print (max_graph_size)
  num_batch = len(batch_molecules)

  adj_ph = np.ones([num_batch, max_graph_size, max_graph_size],
                    dtype=np.float32)*-10.0
  node_feature_ph = np.ones([num_batch, max_graph_size, n_node_types],
                             dtype=np.float32)*(0.05/(n_node_types))
  edge_features_ph = np.ones([num_batch, max_graph_size,
                              max_graph_size, n_edge_types+1],
                              dtype=np.float32)*-2.0
  mask_ph = np.zeros([num_batch, max_graph_size], dtype=np.float32)

  for idx, m in enumerate(batch_molecules):
    edges, node_types = m.bonds, m.atoms
    if permute:
      edges, node_types = mdu.permute_graph(edges, node_types)

    edges, edge_types, node_types, n_edges = mdu.normalize_graph(
        edges, node_types)

    adj_ph[idx][edges[:, 0], edges[:, 1]] = 10.0
    node_feature_ph[idx][np.arange(len(node_types)), node_types] += 0.95
    node_feature_ph[idx] = np.log(node_feature_ph[idx])
    edge_features_ph[idx][edges[:, 0], edges[:, 1], edge_types] = 2.0
    if FLAGS.symmetric:
      edge_features_ph[idx][edges[:, 1], edges[:, 0], edge_types] = 2.0
      adj_ph[idx][edges[:, 1], edges[:, 0]] = 10.0
    mask_ph[idx][np.arange(len(node_types))] = 1.0

  if FLAGS.measure_ROC and FLAGS.only_score:
    # fake_data is a boolean to indicate if the data to be given is a fake
    # set or a real one.
    if fake_data:
      r = np.random.rand(*node_feature_ph.shape)*np.max(node_feature_ph)
      mask = np.random.choice(2,
                              size=node_feature_ph.shape,
                              p=[0.99, 0.01]).astype(np.bool)
      node_feature_ph[mask] = r[mask]
      # Perturb the adjacency matrix too
      r = np.random.uniform(low=-1.0,
                            high=1.0,
                            size=adj_ph.shape)*np.max(adj_ph)
      mask = np.random.choice(2,
                              size=adj_ph.shape,
                              p=[0.99, 0.01]).astype(np.bool)
      adj_ph[mask] = r[mask]
      # Perturb the edge features if required
      if FLAGS.use_edge_features:
        r = np.random.rand(*edge_features_ph.shape)*np.max(edge_features_ph)
        mask = np.random.choice(2,
                                size=edge_features_ph.shape,
                                p=[0.99, 0.01]).astype(np.bool)
        edge_features_ph[mask] = r[mask]

  if FLAGS.perturb_data:
    # Perturb the node features
    r = np.random.rand(*node_feature_ph.shape)*np.max(node_feature_ph)
    mask = np.random.choice(2,
                            size=node_feature_ph.shape,
                            p=[0.8, 0.2]).astype(np.bool)
    node_feature_ph[mask] = r[mask]
    # Perturb the adjacency matrix too
    r = np.random.uniform(low=-1.0,
                          high=1.0,
                          size=adj_ph.shape)*np.max(adj_ph)
    mask = np.random.choice(2,
                            size=adj_ph.shape,
                            p=[0.8, 0.2]).astype(np.bool)
    adj_ph[mask] = r[mask]

  return fill_feed_dict((placeholders['z_in'], adj_ph),
                        (placeholders['omega_in'], node_feature_ph),
                        (placeholders['edge_in'], edge_features_ph),
                        (placeholders['mask_in'], mask_ph))

def build_sample_graph(model, model_hparams):
  placeholders = model.set_inputs()
  log_prob, out, edge_feat = model.setup(is_training=False,
                                         sample=True)
  loss = tf.reduce_mean(log_prob)
  loss_string = tf.summary.scalar('sample_log_prob', loss)
  z, omega = out
  z_string = tf.summary.image('sampled_adj', tf.expand_dims(z, 3))
  omega_string = tf.summary.image('sampled_omega', tf.expand_dims(omega, 3))
  summary_op = tf.summary.merge([loss_string, z_string, omega_string])
  return placeholders, loss, summary_op, (z, omega), edge_feat

def build_train_graph(model, model_hparams):
  placeholders = model.set_inputs()
  if FLAGS.time_dependent_prior:
    assgn_ph, assgn_vars = model.set_assign_placeholders()
    assgn_ops = model.assign_op(assgn_ph)
  train_log_prob = model.setup(is_training=True)
  test_log_prob = model.setup(is_training=False)

  train_loss = tf.reduce_mean(-train_log_prob)
  if not FLAGS.measure_ROC:
    test_loss = tf.reduce_mean(-test_log_prob)
  else:
    test_loss = -test_log_prob

  train_loss_string = tf.summary.scalar("train_loss", train_loss)
  test_loss_string = tf.summary.scalar("test_loss", test_loss)

  # Optimization
  train_variables = tf.trainable_variables()
  _print_trainable_vars()
  loss = train_loss
  if FLAGS.add_weight_decay:
    loss += tf.add_n([model_hparams.weight_decay*tf.nn.l2_loss(var) for var\
                                                      in train_variables])
  loss_string = tf.summary.scalar("total_loss", loss)
  if FLAGS.clip_gradients:
    grads = tf.gradients(loss, train_variables)
    grads, norm = tf.clip_by_global_norm(grads, model_hparams.max_grad_norm)

  global_step = tf.Variable(0, trainable=False)
  learning_rate = FLAGS.learning_rate

  opt = tf.train.AdamOptimizer(learning_rate)
  if FLAGS.clip_gradients:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      update_op = opt.apply_gradients(zip(grads, train_variables),
                                      global_step)
  else:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      update_op = opt.minimize(loss, global_step)

  summary_list = [train_loss_string, loss_string]
  summary_op = tf.summary.merge(summary_list)
  if FLAGS.time_dependent_prior:
    return placeholders, train_loss, test_loss, loss, update_op,\
           global_step, summary_op, assgn_ph, assgn_ops, assgn_vars
  return placeholders, train_loss, test_loss, loss, update_op,\
                                                global_step, summary_op

def train(model_hparams,
          train_data, valid_data, eval_every,
          exp_dir="",
          summary_writer=None,
          n_node_types=None,
          n_edge_types=None):
  assert n_node_types is not None
  assert n_edge_types is not None

  with tf.Graph().as_default():
    params = dict()
    params['is_training'] = True
    params['n_node_types'] = n_node_types
    params['n_edge_types'] = n_edge_types
    model_hparams.add_hparam('node_dim', int(n_node_types/2))
    model_hparams.add_hparam('edge_dim', n_edge_types)
    model = graphgen.GraphGenerator(model_hparams, params, 'graph-gen')
    if FLAGS.time_dependent_prior:
      placeholders, train_loss, valid_loss, overall_loss,\
            update_op, global_step, summ_op, assign_ph, assgn_ops,\
                         assign_vars = build_train_graph(model, model_hparams)
    else:
      placeholders, train_loss, valid_loss, overall_loss,\
       update_op, global_step, summ_op = build_train_graph(model, model_hparams)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver(tf.all_variables())

      ckpt = tf.train.latest_checkpoint(exp_dir)
      if ckpt:
        print ("Reading model parameters from %s" % ckpt)
        saver.restore(sess, ckpt)
      else:
        print ("Unable to load checkpoint")

      gs = sess.run(global_step)
      summary_writer = tf.summary.FileWriter(exp_dir, sess.graph)

      while gs < FLAGS.train_steps:
        feed_dict = get_data_for_pass(train_data, placeholders, True,
                                      n_node_types=n_node_types,
                                      n_edge_types=n_edge_types)
        if not FLAGS.only_score:
          _, loss_train, summ_result = sess.run([update_op, overall_loss,
                                             summ_op], feed_dict)
          print (loss_train)
          summary_writer.add_summary(summ_result, gs)
        gs = int(sess.run(global_step))

        if gs % eval_every == 0 and FLAGS.measure_ROC and FLAGS.only_score:
          log_prob_dataset = []
          for i in range(valid_data._n//FLAGS.batch_size):
            feed_dict = get_data_for_pass(valid_data, placeholders, True,
                                        n_node_types=n_node_types,
                                        n_edge_types=n_edge_types,
                                        fake_data=False)
            loss_validation = sess.run(valid_loss, feed_dict)
            log_prob_dataset.extend(loss_validation)

          # Perturb data to get random data values here
          log_prob_fake = []
          for i in range(valid_data._n//FLAGS.batch_size):
            feed_dict = get_data_for_pass(valid_data, placeholders, True,
                                        n_node_types=n_node_types,
                                        n_edge_types=n_edge_types,
                                        fake_data=True)
            loss_validation = sess.run(valid_loss, feed_dict)
            log_prob_fake.extend(loss_validation)
          log_prob_dataset.sort()
          log_prob_fake.sort()
          print ('Log_Prob: ')
          print ('-----------------------------------------------------')
          print (log_prob_dataset)
          print ('-----------------------------------------------------')
          print (log_prob_fake)
          tuple_list_dataset = [(-x,1) for x in log_prob_dataset]
          tuple_list_fake = [(-x, 0) for x in log_prob_fake]
          tuple_list_dataset.extend(tuple_list_fake)
          tuple_list_dataset.sort(key=lambda elem: elem[0])

          num_examples = len(tuple_list_dataset)
          tuple_list_dataset = np.array(tuple_list_dataset)
          scores = tuple_list_dataset[:, 0]
          labels = tuple_list_dataset[:, 1]
          fpr, tpr, thresholds = metrics.roc_curve(labels, scores,
                                                   drop_intermediate=True)
          print ('False positive Rate: ')
          print (fpr.tolist())
          print ('True positive Rate: ')
          print (tpr.tolist())
          print ('Thresholds: ')
          print (thresholds.tolist())
          return

        if gs % eval_every == 0:
          if exp_dir:
            checkpoint_path = os.path.join(exp_dir, "model.ckpt")
            saver.save(sess, checkpoint_path, global_step=global_step)

          avg_loss = 0.0
          avg_loss_per_dim = 0.0
          for i in range(valid_data._n//FLAGS.batch_size):
            feed_dict = get_data_for_pass(valid_data, placeholders, True,
                                        n_node_types=n_node_types,
                                        n_edge_types=n_edge_types)
            loss_validation = sess.run(valid_loss, feed_dict)
            avg_loss += loss_validation
            print ('-------------------------------------------------------')
            print ('Loss Validation : ', loss_validation)
            if i == 0:
              print ('-------------------------------------------------------')
          avg_loss /= (valid_data._n//FLAGS.batch_size)
          print ('============================================================')
          print ('Avg Loss Validation: ', avg_loss)
          print ('============================================================')
          summ_val = tf.Summary(value=[
            tf.Summary.Value(tag='loss_test', simple_value=avg_loss)])
          summary_writer.add_summary(summ_val, gs)

          if gs % FLAGS.sharpen_steps == 0 and FLAGS.time_dependent_prior:
            increment_factor = 1.02
            assign_vars_val = sess.run(assign_vars, feed_dict=())
            print ('Value before update: ', assign_vars_val)
            assign_vars_val = [var_val*increment_factor for var_val in
                               assign_vars_val]
            assign_ph_list = [assign_ph['omega_alpha'], assign_ph['omega_beta'],
                              assign_ph['z_alpha'], assign_ph['z_beta']]
            if FLAGS.use_edge_features:
              assign_ph_list.extend([assign_ph['ef_alpha'],
                                     assign_ph['ef_beta']])
            feed_dict = dict(zip(assign_ph_list, assign_vars_val))
            sess.run(assgn_ops, feed_dict)
            assign_vars_val = sess.run(assign_vars, feed_dict=())
            print ('Value after update: ', assign_vars_val)

def sample(model_hparams,
          train_data, valid_data, eval_every,
          exp_dir="",
          summary_writer=None,
          n_node_types=None,
          n_edge_types=None):
  assert n_node_types is not None
  assert n_edge_types is not None

  with tf.Graph().as_default():
    params = dict()
    params['is_training'] = True
    params['n_node_types'] = n_node_types
    params['n_edge_types'] = n_edge_types
    model_hparams.add_hparam('node_dim', int(n_node_types/2))
    model_hparams.add_hparam('edge_dim', n_edge_types)
    model = graphgen.GraphGenerator(model_hparams, params, 'graph-gen')
    placeholders, loss, summ_op, out_sample, edge_feat = build_sample_graph(
        model, model_hparams)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver(tf.all_variables())
      ckpt = tf.train.latest_checkpoint(exp_dir)
      if ckpt:
        print ("Reading model parameters from %s" % ckpt)
        saver.restore(sess, ckpt)
      else:
        print ("Unable to load checkpoint")
      summary_writer = tf.summary.FileWriter(exp_dir, sess.graph)

      # Take out num_samples samples at a time.
      for i in range(FLAGS.num_samples):
        num_nodes = np.random.randint(low=25, high=36)
        z_in = np.random.beta(a=4.0, b=8.0, size=(num_nodes, num_nodes))
        z_in_up = np.triu(z_in, k=1)
        z_in_new = z_in_up + np.transpose(z_in_up)
        z_in = z_in_new + np.identity(num_nodes)*z_in
        z_in = np.expand_dims(z_in, 0)
        # to account for the transformation to z
        z_in = np.log(z_in + 1e-10) - np.log(1 - z_in + 1e-10)
        print ('z_in: ', z_in[0, 0, 0:10])

        edge_in = np.random.beta(a=4.0, b=8.0,
                                 size=(num_nodes, num_nodes, n_edge_types+1))
        edge_in_up = edge_in * np.expand_dims(np.triu(
            np.ones((num_nodes, num_nodes)), k=1), 2)
        edge_in_new = edge_in_up + np.transpose(edge_in_up, (1,0,2))
        edge_in = edge_in_new + np.expand_dims(np.identity(num_nodes), 2)*edge_in
        edge_in = np.expand_dims(edge_in, 0)
        edge_in = np.log(edge_in+1e-10) - np.log(1-edge_in + 1e-10)

        omega_in = np.random.beta(a=4.0, b=8.0,
                                  size=(num_nodes, 2*model_hparams.node_dim))
        omega_in = np.log(omega_in + 1e-10) - np.log(1- omega_in + 1e-10)
        print ('omega_in: ', omega_in[0, 0:10])
        mask_in = np.ones((1, num_nodes))
        omega_in = np.expand_dims(omega_in, 0)
        feed_dict = fill_feed_dict((placeholders['z_in'], z_in),
                                   (placeholders['omega_in'], omega_in),
                                   (placeholders['mask_in'], mask_in),
                                   (placeholders['edge_in'], edge_in))
        log_prob, summ_out, sampled_output, edge_val = sess.run(
            [loss, summ_op, out_sample, edge_feat], feed_dict)
        summary_writer.add_summary(summ_out, i)
        print ('Log Prob: ', log_prob)
        print ('Adjacency matrix: ')
        print ('---------------------------------------')
        _print_matrix_function(sampled_output[0][0])
        print ('---------------------------------------')
        print ('Node features: ')
        print ('---------------------------------------')
        _print_matrix_function(sampled_output[1][0])
        print ('---------------------------------------')
        '''print ('Edge features: ')
        print ('---------------------------------------')
        _print_matrix_function(edge_val[0])'''
        print ('=======================================')

def main(argv):
  tf.set_random_seed(FLAGS.seed)
  if not tf.gfile.IsDirectory(FLAGS.exp_dir):
    tf.gfile.MakeDirs(FLAGS.exp_dir)

  print ('Made directory')
  train_set, val_set, _ = mdu.read_dataset(FLAGS.dataset)
  molecule_mapping = mdu.read_molecule_mapping_for_set(FLAGS.dataset)
  inv_mol_mapping = {v: k for k, v in enumerate(molecule_mapping)}

  if FLAGS.dataset.startswith('zinc'):
    bond_mapping = mdu.read_bond_mapping_for_set(FLAGS.dataset)
    inv_bond_mapping = {('%d_%d' % v): k for k, v in enumerate(bond_mapping)}
    stereo = True
  else:
    bond_mapping = None
    inv_bond_mapping = None
    stereo = False

  # unique set of training data, used for evaluation
  train_set_unique = set(train_set)
  train_set, val_set, _ = mdu.read_molecule_graphs_set(FLAGS.dataset)

  n_node_types = len(molecule_mapping)
  print (n_node_types)
  n_edge_types = mdu.get_max_edge_type(train_set) + 1
  max_n_nodes = max(len(m.atoms) for m in train_set)

  train_set = mdu.Dataset(
      train_set, FLAGS.batch_size, shuffle=True)
  val_set = mdu.Dataset(
      val_set, FLAGS.batch_size, shuffle=True)

  # n_node_types: number of node types (assumed categorical)
  # n_edge_types: number of edge types/ labels
  model_hparams = hparams.get_hparams_ChEMBL()
  print ('Number of node/edge types: ', n_node_types, n_edge_types)
  print ('Inside train function now...')
  with tf.device('/gpu:1'):
    if FLAGS.sample:
      sample(model_hparams, train_set, val_set, eval_every=FLAGS.eval_every,
             exp_dir=FLAGS.exp_dir,
             summary_writer=None,
             n_node_types=n_node_types,
             n_edge_types=n_edge_types)
    else:
      train(model_hparams, train_set, val_set, eval_every=FLAGS.eval_every,
            exp_dir=FLAGS.exp_dir,
            summary_writer=None,
            n_node_types=n_node_types,
            n_edge_types=n_edge_types)

if __name__ == '__main__':
  tf.app.run()

