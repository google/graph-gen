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

"""Train a Reversible GNN on Citation Networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils # TBD
import lbfgs # TBD
import metrics # TBD
from graphNN import citation_model
from hparams import get_hparams
import preprocess_ppi # TBD

import tensorflow as tf
import numpy as np
import scipy.sparse as sparse
import collections, os, time
import random
from sklearn import decomposition

flags = tf.app.flags
flags.DEFINE_enum('dataset', 'cora', ['cora', 'citeseer', 'pubmed', 'ppi'],
                  'Name of the dataset to test on')
flags.DEFINE_integer('batch_size', 20, 'Batch size for training')
flags.DEFINE_integer('eval_every', 100,
                     'Number of epochs after which to evaluate')
flags.DEFINE_string('exp_dir', './citation_net_rev_mpnn/',
                    'Directory where to store results and checkpoints')
flags.DEFINE_bool('clip_gradients', False, 'Whether to clip gradients or not')
flags.DEFINE_integer('train_steps', 200, 'Number of steps to train the model')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate to be used for training')
flags.DEFINE_string('data_dir', '',
                    'Data directory for the dataset')
flags.DEFINE_bool('test_regularly', True,
                  'Flag to specify if we should do testing after eval_every epochs')
flags.DEFINE_bool('add_weight_decay', True,
                  'Flag to set if we want to add weight decay or not')
flags.DEFINE_bool('reversible', False,
                  'Flag to set if reversibile GNN is to be run or not')
flags.DEFINE_bool('neumann_rbp', False,
                  'Flag to set if neumann_rbp is to be run or not')
flags.DEFINE_integer('seed', 1234,
                     'Random Seed which will be used by tensorflow.')
flags.DEFINE_enum('mode', 'perf', ['perf', 'diagnose'],
                  'Name of the mode to run (accuracy/diagnostic)')
flags.DEFINE_bool('reduce_dim', False,
               'Flag to specify whether dimensions should be reduced for exp.')
flags.DEFINE_bool('use_random_labels', False,
                  'Whether to use random labels or not for testing memorization.')
flags.DEFINE_bool('use_lbfgs', False,
                  'Whether to use L-BFGS optimizer for experiments.')
flags.DEFINE_bool('use_inp_noise', False,
                  'Whether to use input noise on the input data.')
flags.DEFINE_bool('use_linear_transform', False,
                  'Whether to use linear transforms to reduce dimensionality.')
flags.DEFINE_bool('extra_noise_dim', False,
                  'Whether to add extra dimensions just containing noise.')
flags.DEFINE_bool('batch_norm', False,
                  'Whether to use batch norm or not for training the'\
                  'classification part, to prevent shoot up of losses.')
flags.DEFINE_bool('bag_of_nodes', False,
                  'Whether to use a bag of nodes model for classification')
FLAGS = flags.FLAGS

PERM = np.random.permutation(2708)


def preprocess_features(features):
  """Row-normalize feature matrix and convert to tuple representation."""
  rowsum = np.array(features.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sparse.diags(r_inv, 0)
  features = r_mat_inv.dot(features)
  return features

def _print_trainable_vars():
  print ('----------------------------------')
  print ('Trainable/All Variables: ')
  for x in tf.trainable_variables():
    print (x)
  print ('----------------------------------')

def construct_feed_dict(features, labels, label_mask, adj_mat,
                        placeholders, indx=None, batch_size=None,
                       perm_temp=None, noise=False):
  """Construct a feed_dict dictionary for train/eval purposes."""
  if FLAGS.dataset == "ppi":
    assert indx is not None
    assert batch_size is not None
    feed_dict = dict()
    if perm_temp is None:
      perm_temp = np.random.permutation(labels.shape[0])
      perm_temp = np.arange(labels.shape[0])
    feed_dict[placeholders['labels']] = labels[perm_temp][indx*batch_size:\
                                               (indx+1)*batch_size]
    if FLAGS.use_random_labels:
      print (indx)
      feed_dict[placeholders['labels']] = feed_dict[placeholders[
          'labels']][PERM[indx]]
    feed_dict[placeholders['label_mask']] = label_mask[perm_temp][indx*batch_size:\
                                               (indx+1)*batch_size]
    feed_dict[placeholders['state']] = features[perm_temp][indx*batch_size:\
                                               (indx+1)*batch_size]
    feed_dict[placeholders['adjacency']] = adj_mat[perm_temp][indx*batch_size:\
                                               (indx+1)*batch_size]
  else:
    feed_dict = dict()
    feed_dict[placeholders['labels']] = np.expand_dims(labels, 0)
    if FLAGS.use_random_labels:
      feed_dict[placeholders['labels']] = np.expand_dims(labels[PERM], 0)
    feed_dict[placeholders['label_mask']] = np.expand_dims(label_mask, 0)
    feed_dict[placeholders['state']] = np.expand_dims(features, 0)
    feed_dict[placeholders['adjacency']] = np.expand_dims(adj_mat, 0)

  if FLAGS.use_inp_noise and noise:
    temp_shape = feed_dict[placeholders['state']].shape
    feed_dict[placeholders['state']] += np.random.normal(0.0, 0.1, temp_shape)

  if FLAGS.extra_noise_dim:
    orig_states = feed_dict[placeholders['state']]
    feed_dict[placeholders['state']] = np.concatenate([orig_states,
        np.random.normal(0.0, 0.1, (orig_states.shape[0],
                                    orig_states.shape[1], 300))], 2)
  return feed_dict

def get_metrics(outputs, labels, label_mask):
  """Return loss/F1/Accuracy metrics for evaluation of the model."""
  if FLAGS.dataset == "ppi":
    # account for batch size and the loss
    # outputs: [b x n x L]
    labels = tf.cast(labels, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs,
                                                   labels=labels)
    loss = tf.reduce_mean(loss, axis=2)
    label_mask = tf.cast(label_mask, dtype=tf.float32)
    label_mask /= tf.reduce_mean(label_mask)
    loss *= label_mask
    loss = tf.reduce_mean(loss)
    preds = tf.nn.sigmoid(outputs)
    out_labels = tf.cast((preds > 0.5), tf.float32)
    correct_prediction = tf.equal(out_labels, labels)
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    accuracy_all *= tf.expand_dims(label_mask, 2)
    accuracy = tf.reduce_mean(accuracy_all)

    # More important F1 score
    tp = tf.cast(tf.count_nonzero(out_labels*labels), tf.float32)
    tn = tf.cast(tf.count_nonzero((out_labels -1)*(labels - 1)), tf.float32)
    fp = tf.cast(tf.count_nonzero(out_labels * (labels - 1)), tf.float32)
    fn = tf.cast(tf.count_nonzero((out_labels - 1) * labels), tf.float32)
    precision = tp/(tp + fp)
    recall = tp/ (tp +fn)
    f1_score = 2.0*precision*recall/(precision+recall)
    return {'loss': loss,
            'accuracy': accuracy,
            'output_labels': preds,
            'precision': precision,
            'recall' : recall,
            'f1': f1_score}
  outputs = tf.squeeze(outputs, 0)
  labels = tf.squeeze(labels, 0)
  label_mask = tf.squeeze(label_mask, 0)
  loss = metrics.masked_softmax_cross_entropy(outputs, labels, label_mask)
  accuracy = metrics.masked_accuracy(outputs, labels, label_mask)
  confusion_matrix = metrics.masked_confusion_matrix(outputs,
                                                     labels,
                                                     label_mask)
  predicted_output = tf.nn.softmax(outputs)
  return {'loss': loss,
          'accuracy': accuracy,
          'confusion_matrix': confusion_matrix,
          'output_labels': predicted_output}


def build_train_graph(model_hparams, model):
  """Build the train graph for the model."""
  placeholders = model.get_inputs()
  summary_op = None
  train_pred = model.setup(is_training=True)
  test_pred = model.setup(is_training=False)
  train_logits = train_pred['logits']
  test_logits = test_pred['logits']

  target_labels = tf.placeholder(tf.float32,
                                 [None, None, model_hparams.num_classes])
  target_mask = tf.placeholder(tf.int32)

  placeholders['labels'] = target_labels
  placeholders['label_mask'] = target_mask

  train_metrics = get_metrics(train_logits, target_labels, target_mask)
  test_metrics = get_metrics(test_logits, target_labels, target_mask)
  loss = train_metrics['loss']

  # Set up summaries
  train_loss_string = tf.summary.scalar("train_loss", loss)
  # test_loss_string = tf.summary.scalar("test_loss", test_metrics['loss'])
  train_acc_string = tf.summary.scalar("train_acc", train_metrics['accuracy'])
  # test_acc_string = tf.summary.scalar("test_acc", test_metrics['accuracy'])
  if FLAGS.dataset == "ppi":
    train_f1_string = tf.summary.scalar("train_f1", train_metrics['f1'])
    train_p_string = tf.summary.scalar("train_p", train_metrics['precision'])
    train_r_string = tf.summary.scalar("train_r", train_metrics['recall'])

  trainable_variables = tf.trainable_variables()

  if FLAGS.mode == "diagnose":
    trainable_variables = [var_idx for  var_idx in trainable_variables if\
                           ("output_fn" in var_idx.name and "neumann_mp" not\
                            in var_idx.name)]
  if FLAGS.add_weight_decay:
    loss += tf.add_n([model_hparams.weight_decay*tf.nn.l2_loss(var) for var\
                                                      in trainable_variables])

  _print_trainable_vars()

  if FLAGS.clip_gradients or FLAGS.use_lbfgs:
    grads = tf.gradients(loss, trainable_variables)
    if not FLAGS.use_lbfgs:
      grads, norm = tf.clip_by_global_norm(grads, model_hparams.max_grad_norm)

  global_step = tf.Variable(0, trainable=False)
  learning_rate = FLAGS.learning_rate

  if FLAGS.use_lbfgs:
    opt = lbfgs.LBFGS(trainable_variables, grads, learning_rate,
                     parameters_save=trainable_variables,
                     logging=True)
  elif FLAGS.neumann_rbp:
    opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt = tf.train.AdamOptimizer(learning_rate)
  else:
    opt = tf.train.AdamOptimizer(learning_rate)

  if FLAGS.use_lbfgs:
    update_op, reset_op = opt.components()
  elif FLAGS.clip_gradients:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      update_op = opt.apply_gradients(zip(grads, trainable_variables),
                                      global_step)
  else:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      update_op = opt.minimize(loss, global_step)

  summary_list = [train_loss_string,train_acc_string]
  if FLAGS.dataset == "ppi":
    summary_list += [train_f1_string, train_p_string, train_r_string]
  summary_op = tf.summary.merge(summary_list)
  return placeholders, train_metrics, test_metrics, update_op, global_step, summary_op


def train(model_hparams,
          eval_every=5,
          exp_dir="",
          summary_writer=None):
  """Run training for the model on Cora/Pubmed/PPI datasets."""
  if FLAGS.dataset == "ppi":
    dataset = preprocess_ppi.process_p2p(FLAGS.data_dir)
    train_data = dataset[0]
    valid_data = dataset[1]
    test_data = dataset[2]
    train_data_features = train_data.features
    valid_data_features = valid_data.features
    test_data_features = test_data.features
  else:
    dataset = utils.load_data(FLAGS.dataset, FLAGS.data_dir)
    features = preprocess_features(dataset.features)

  params = dict()
  if FLAGS.neumann_rbp:
    model = citation_model.NeumannMPNN(params, model_hparams)
  elif FLAGS.reversible:
    model = citation_model.RevMPNN(params, model_hparams)
  else:
    model = citation_model.VanillaMPNN(params, model_hparams)
  model.set_inputs()
  placeholders, train_metrics, test_metrics, update_op,\
                  global_step, summary_op = build_train_graph(model_hparams,
                                                              model)
  train_loss, test_loss = train_metrics['loss'], test_metrics['loss']
  accuracy, test_accuracy = train_metrics['accuracy'], test_metrics['accuracy']
  if FLAGS.dataset == "ppi":
    f1_train, f1_test = train_metrics['f1'], test_metrics['f1']
    prec_train, prec_test = train_metrics['precision'], test_metrics['precision']
    rec_train, rec_test = train_metrics['recall'], test_metrics['recall']
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

  if "ppi" not in FLAGS.dataset:
    features = sparse.csr_matrix.todense(features)
    adj_mat = sparse.csr_matrix.todense(dataset.adj)
  if FLAGS.reduce_dim:
    if FLAGS.dataset == "ppi":
      train_data_temp = np.reshape(train_data.features, [-1, 50])
      valid_data_temp = np.reshape(valid_data.features, [-1, 50])
      test_data_temp = np.reshape(test_data.features, [-1, 50])
      overall_features = np.concatenate([train_data_temp, valid_data_temp,
                                         test_data_temp], 0)
      features = overall_features
    pca = decomposition.PCA(n_components=model_hparams.node_dim)
    features = pca.fit_transform(features)

  if "ppi" not in FLAGS.dataset:
    train_feed_dict = construct_feed_dict(features,
                                          dataset.labels,
                                          dataset.train_mask,
                                          adj_mat,
                                          placeholders, noise=True)
    val_feed_dict = construct_feed_dict(features, dataset.labels,
                                  dataset.val_mask,
                                  adj_mat,
                                  placeholders, noise=False)
    test_feed_dict = construct_feed_dict(features, dataset.labels,
                                  dataset.test_mask,
                                  adj_mat,
                                  placeholders, noise=False)
  if FLAGS.dataset == "ppi" and FLAGS.reduce_dim:
    print ('Reduced the dimensions')
    train_data_features = np.reshape(pca.transform(train_data_temp),
                     [train_data.labels.shape[0], -1, model_hparams.node_dim])
    valid_data_features = np.reshape(pca.transform(valid_data_temp),
                     [valid_data.labels.shape[0], -1, model_hparams.node_dim])
    test_data_features = np.reshape(pca.transform(test_data_temp),
                     [test_data.labels.shape[0], -1, model_hparams.node_dim])
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.all_variables())
  summary_writer = tf.summary.FileWriter(FLAGS.exp_dir, sess.graph)

  ckpt = tf.train.latest_checkpoint(exp_dir)
  if ckpt:
    print ("Reading Model parameters from %s" % ckpt)
    saver.restore(sess, ckpt)
  else:
    print ("Unable to load checkpoint.")

  gs = sess.run(global_step)
  indx = 0
  if FLAGS.dataset == "ppi":
    perm_temp = np.random.permutation(train_data.features.shape[0])

  while gs < FLAGS.train_steps:
    eval_list = [update_op, train_loss, accuracy]
    if summary_writer is not None:
      eval_list += [summary_op]
    if FLAGS.dataset == "ppi":
      train_feed_dict = construct_feed_dict(train_data_features,
                                            train_data.labels,
                                            train_data.mask,
                                            train_data.adj,
                                            placeholders,
                                            indx, FLAGS.batch_size)
      indx = (indx + 1)%(20 // FLAGS.batch_size)
      if indx == 0:
        perm_temp = np.random.permutation(train_data.features.shape[0])
        print (indx, indx // FLAGS.batch_size)
    if FLAGS.use_inp_noise:
      train_feed_dict = construct_feed_dict(features,
                                            dataset.labels,
                                            dataset.train_mask,
                                            adj_mat,
                                            placeholders, noise=True)
    eval_out = sess.run(eval_list, train_feed_dict)
    loss_i, train_acc = eval_out[1], eval_out[2]
    summ = None
    print ('Train Loss: ', loss_i, ' Train_acc: ', train_acc)
    if len(eval_out) > 3:
      summ = eval_out[3]

    if summary_writer is not None:
      summary_writer.add_summary(summ, gs)

    if not FLAGS.use_lbfgs:
      gs = int(sess.run(global_step))
    else:
      gs += 1

    if gs % eval_every == 0:
      if exp_dir:
        checkpoint_path = os.path.join(exp_dir, "model.ckpt")
        saver.save(sess, checkpoint_path, global_step)
      if FLAGS.dataset == "ppi":
        val_feed_dict = construct_feed_dict(valid_data_features,
                                            valid_data.labels,
                                            valid_data.mask,
                                            valid_data.adj,
                                            placeholders, 0, 2)
      if FLAGS.dataset == "ppi":
        cost_val, acc_val, val_f1, val_prec,\
                      val_recall = sess.run([test_loss, test_accuracy,
                                            f1_test, prec_test, rec_test],
                                            val_feed_dict)
      else:
        cost_val, acc_val = sess.run([test_loss, test_accuracy], val_feed_dict)
      print ('=============================================')
      if FLAGS.dataset == "ppi":
        print ('Eval Loss: ', cost_val, 'Eval Acc: ', acc_val, 'F1: ', val_f1,
               'Prec: ', val_prec, 'Rec: ', val_recall)
      else:
        print ('Eval Loss: ', cost_val, 'Eval Acc: ', acc_val)
      if summary_writer is not None:
        summ_val = tf.Summary(value=[
            tf.Summary.Value(tag='val_acc', simple_value=acc_val),
            tf.Summary.Value(tag='val_cost', simple_value=cost_val)])
        summary_writer.add_summary(summ_val, gs)

      if FLAGS.test_regularly:
        if FLAGS.dataset == "ppi":
          test_feed_dict = construct_feed_dict(test_data_features,
                                               test_data.labels,
                                               test_data.mask,
                                               test_data.adj,
                                                placeholders, 0, 2)
        if FLAGS.dataset == "ppi":
          cost_test, acc_test, test_f1, test_prec,\
                      test_recall = sess.run([test_loss, test_accuracy,
                                              f1_test, prec_test, rec_test],
                                              test_feed_dict)
          print ('Test Loss: ', cost_test, 'Test Acc: ', acc_test, 'F1: ',
                 test_f1, 'Prec: ', test_prec, 'Rec: ', test_recall)
        else:
          cost_test, acc_test = sess.run([test_loss, test_accuracy],
                                          test_feed_dict)
          print ('Test Loss: ', cost_test, 'Test Acc: ', acc_test)

        if summary_writer is not None:
          summ_test = tf.Summary(value=[
              tf.Summary.Value(tag='test_acc', simple_value=acc_test),
              tf.Summary.Value(tag='test_cost', simple_value=cost_test)])
          summary_writer.add_summary(summ_test, gs)
      print ('=============================================')


def main(argv):
  tf.set_random_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  if not os.path.exists(FLAGS.exp_dir):
    os.makedirs(FLAGS.exp_dir)

  if FLAGS.dataset == "ppi":
    model_hparams = get_hparams('PPI')
  else:
    model_hparams = get_hparams('CitationNet')
  with tf.device('/gpu:1'):
      train(model_hparams, FLAGS.eval_every, FLAGS.exp_dir, None, None)


if __name__ == '__main__':
  app.run(main)
