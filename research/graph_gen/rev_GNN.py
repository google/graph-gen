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
"""Reversible Graph Neural Network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
import tensorflow as tf
import numpy as np
import re

flags = tf.app.flags
flags.DEFINE_bool('use_decay', False,
                  'Whether to use decay or not in the update equations.')
flags.DEFINE_bool('use_sigmoid', False,
                  'Whether to use sigmoid transformation over the sum/can'\
                  'be replaced by any bijective non-linearity')
flags.DEFINE_enum('function_type', 'sigmoid', ['sigmoid', 'tanh'],
                  'Non-linearity to be applied to the system.')
flags.DEFINE_bool('dampen_grad', False,
                  'Whether to dampen the gradients passing on to the variables')
FLAGS = flags.FLAGS

LAYER_RE = re.compile(".*rev_mp/layer_([0-9]*)/([fgb])/.*")

def _print_tensors(*args):
  """Print tensor shapes while graph construction."""
  print ('Printing Tensors =====================')
  for idxs, arg in enumerate(args):
    if isinstance(arg, list):
      for idx, val in enumerate(arg):
        print (val)
    else:
      print (arg)
  print ('End Printing Tensors ==================')


def _get_variable_util(f, g, variables, num_layers):
  """Get back the variable list from scopes.
     Used from tensor2tensor revnet block."""
  f_vars = []
  g_vars = []
  vars_in_scope = variables
  var_name_dict = dict()
  id_cntr = 0

  # Assumes that parameters across layers are shared
  for i, var in enumerate(vars_in_scope):
    print (var.name)
    regex = LAYER_RE.match(var.name)
    layer_num = int(regex.group(1))
    fn_name = regex.group(2)

    if var.name not in var_name_dict:
      var_name_dict[var.name] = id_cntr
      id_cntr += 1

    if fn_name == 'f' or fn_name == 'b':
      f_vars.append(var)
    if fn_name == 'g' or fn_name == 'b':
      g_vars.append(var)

  return f_vars, g_vars, var_name_dict

def message_passing_step(inputs,
                            msg_fn,
                            agg_fn,
                            state_fn,
                            *args):
  """Vanilla Message passing step implementing f/g."""
  num_nodes = args[0]
  adj_mat = args[1]
  dims_for_state_fn = args[2]
  incr_node_rep = inputs

  for i in range(1):
    # Message passing code here
    init_shape = tf.shape(incr_node_rep)
    temp_t = tf.tile(tf.expand_dims(incr_node_rep,2),
                                      [1, 1, init_shape[1], 1])

    m_t = msg_fn(temp_t)
    incoming_node = agg_fn(m_t*tf.expand_dims(adj_mat, 3))
    incr_node_rep = state_fn(incr_node_rep, incoming_node,
                             {'dim': dims_for_state_fn})

  return incr_node_rep


def _mp_forward(xs, f, g, parity, lower_triangular_op=False,
                lambda_t=0):
  """One layer of message passing."""
  h0, h1 = xs
  non_lin_fn = None
  if FLAGS.function_type == "sigmoid":
    non_lin_fn = tf.nn.sigmoid
  elif FLAGS.function_type == "tanh":
    non_lin_fn = tf.nn.tanh
  if parity:
    if FLAGS.use_decay:
      h0_next = f(h1) + h0*lambda_t
    elif FLAGS.use_sigmoid:
      h0_next = non_lin_fn(f(h1) + h0)
    else:
      h0_next = f(h1) + h0
    h1_next = h1
  else:
    h0_next = h0
    if FLAGS.use_decay:
      h1_next = g(h0) + h1*lambda_t
    elif FLAGS.use_sigmoid:
      h1_next = non_lin_fn(g(h0) + h1)
    else:
      h1_next = h1 + g(h0)

  return (h0_next, h1_next)

def _mp_backward(ys, grad_ys, f, g, parity, f_vars, g_vars, lambda_t=0):
  """Backprop operation for one layer."""
  y0, y1 = ys
  grad_y0, grad_y1 = grad_ys

  # Compute the parameter values for the previous step
  if parity:
    x1 = y1
    y1_stop = tf.stop_gradient(y1)
    x1_stop = tf.stop_gradient(x1)
    fy1 = f(y1_stop)
    if FLAGS.use_decay:
      x0 = (y0 - fy1)/(lambda_t)
    elif FLAGS.use_sigmoid:
      y0_stop = tf.stop_gradient(y0)
      if FLAGS.function_type == "tanh":
        x0 = 0.5*(tf.log(1 + y0_stop) - tf.log(1 - y0_stop)) - fy1
      elif FLAGS.function_type == "sigmoid":
        x0 = tf.log(y0_stop + 1e-10) - tf.log(1 - y0_stop + 1e-10) - fy1
    else:
      x0 = y0 - fy1
  else:
    x0 = y0
    y0_stop = tf.stop_gradient(y0)
    x0_stop = tf.stop_gradient(x0)
    gy0 = g(y0_stop)
    if FLAGS.use_decay:
      x1 = (y1 - gy0)/(lambda_t)
    elif FLAGS.use_sigmoid:
      y1_stop = tf.stop_gradient(y1)
      if FLAGS.function_type == "tanh":
        x1 = 0.5*(tf.log(1 + y1_stop) - tf.log(1 - y1_stop)) - gy0
      else:
        x1 = tf.log(y1_stop + 1e-10) - tf.log(1 - y1_stop + 1e-10) - gy0
    else:
      x1 = y1 - gy0

  # Compute the gradients with respect to x0, x1, ws
  retval = [None]
  non_lin_fn = None
  if FLAGS.function_type == "sigmoid":
    non_lin_fn = tf.nn.sigmoid
  elif FLAGS.function_type == "tanh":
    non_lin_fn = tf.nn.tanh

  if parity:
    grad_fy1 = tf.gradients(fy1, y1_stop, grad_y0)[0]
    if FLAGS.use_decay:
      grad_x0 = lambda_t*grad_y0
      grad_x1 = grad_y1 + grad_fy1
      grad_w_f = [gr for gr in tf.gradients(fy1, f_vars, grad_y0)]
    elif FLAGS.use_sigmoid:
      print ('Y0 stop: ', y0_stop, grad_y0)
      temp_y0 = tf.stop_gradient(x0) + tf.stop_gradient(fy1)
      grad_x0 = tf.gradients(non_lin_fn(temp_y0), temp_y0, grad_y0)[0]
      grad_x1 = grad_y1 + tf.gradients(fy1, y1_stop, grad_x0)[0]
      grad_w_f = tf.gradients(fy1, f_vars, grad_x0)
    else:
      grad_x0 = grad_y0
      grad_x1 = grad_y1 + grad_fy1
      grad_w_f = tf.gradients(fy1, f_vars, grad_y0)
    retval = [(x0, x1), (grad_x0, grad_x1), grad_w_f]
  else:
    grad_gy0 = tf.gradients(gy0, y0_stop, grad_y1)[0]
    if FLAGS.use_decay:
      grad_x1 = lambda_t*grad_y1
      grad_x0 = grad_y0 + grad_gy0
      grad_w_g = [gr for gr in tf.gradients(gy0, g_vars, grad_y1)]
    elif FLAGS.use_sigmoid:
      temp_y1 = tf.stop_gradient(x1) + tf.stop_gradient(gy0)
      grad_x1 = tf.gradients(non_lin_fn(temp_y1), temp_y1, grad_y1)[0]
      grad_x0 = grad_y0 + tf.gradients(gy0, y0_stop, grad_x1)
      grad_w_g = tf.gradients(gy0, g_vars, grad_x1)
    else:
      grad_x1 = grad_y1
      grad_x0 = grad_y0 + grad_gy0
      grad_w_g = tf.gradients(gy0, g_vars, grad_y1)
    retval = [(x0, x1), (grad_x0, grad_x1), grad_w_g]

  retval_t = tf.tuple(tf.contrib.framework.nest.flatten(retval))
  retval_tupled = tf.contrib.framework.nest.pack_sequence_as(retval, retval_t)
  return retval

def _rev_mp_block_forward(x0, x1, f, g, num_layers=1):
  """Forward computation for a series of layers."""
  out = (x0, x1)
  # Perform f step once and g step once (this comprises one message passing
  # step in the system.
  lambda_series = [1.0/np.sqrt(cnt+1) for cnt in range(2*num_layers)]
  for i in range(num_layers):
    prev_out = out
    out = _mp_forward(out, f[i], g[i], 1, lambda_t=lambda_series[2*i])
    out = _mp_forward(out, f[i], g[i], 0, lambda_t=lambda_series[2*i+1])

  y0, y1 = out
  return y0, y1


class RevMessagePassingBlock(object):
  """Block to perform reversible message passing."""

  def __init__(self,
               f,
               g,
               num_layers=1,
               is_training=True,
               use_efficient_backprop=True):

    if isinstance(f, list):
      assert len(f) == num_layers
    else:
      f = [f]* num_layers

    if isinstance(g, list):
      assert len(g) == num_layers
    else:
      g = [g]*num_layers

    scope_prefix = "rev_mp/layer_%d/"
    f_scope = scope_prefix + "f"
    g_scope = scope_prefix + "g"

    self.f = f
    self.g = g

    self.num_layers = num_layers
    self.is_training = is_training

    self._use_efficient_backprop = use_efficient_backprop

  def _efficient_grad_fn(self, inputs,
                         variables,
                         ys,
                         grad_ys):
    """Computes gradient for a block of rev GNN layers."""
    f_vars, g_vars, var_names = _get_variable_util(self.f,
                                      self.g, variables, self.num_layers)

    _print_tensors(grad_ys)
    f_var_grads = []
    g_var_grads = []

    # Reversing essential while gradient computation
    f = list(self.f)
    g = list(self.g)
    f.reverse()
    g.reverse()

    lambda_series = [1.0/(np.sqrt(cnt+1)) for cnt in range(2*self.num_layers)]
    lambda_series.reverse()
    for i in range(self.num_layers):
      ys, grad_ys, grad_w_g = _mp_backward(ys, grad_ys, f[i], g[i],
                              0, f_vars,  g_vars, lambda_t=lambda_series[2*i])
      ys, grad_ys, grad_w_f = _mp_backward(ys, grad_ys, f[i], g[i],
                              1, f_vars, g_vars, lambda_t=lambda_series[2*i+1])
      g_var_grads.append(grad_w_g)
      f_var_grads.append(grad_w_f)

    # Reverse variable grads: as variable utility outputs reverse
    f_var_grads.reverse()
    g_var_grads.reverse()
    variable_grads = [None]*len(variables)
    tmp_cntr = 0
    variable_mappings = dict()
    for idx, v in enumerate(variables):
      variable_mappings[v.name] = idx

    # grad_w_g = [variables]
    # Assumption: all variables are present at all time steps
    num_vars_f = len(f_var_grads[0])
    assert num_vars_f == len(f_var_grads[1]), "Number of variables in f"
    num_vars_g = len(g_var_grads[0])
    assert num_vars_g == len(g_var_grads[1]), "Number of variables in g"

    for idxs, values in enumerate(f_var_grads):
      for var_t, grad in list(zip(f_vars, values)):
        indx = variable_mappings[var_t.name]
        if isinstance(grad, tf.IndexedSlices):
          grad = tf.convert_to_tensor(grad)
        variable_grads[indx] = (variable_grads[indx] + grad
                        if variable_grads[indx] is not None else grad if
                        grad is not None else variable_grads[idx])

    for idxs, values in enumerate(g_var_grads):
      for var_t, grad in list(zip(g_vars, values)):
        indx = variable_mappings[var_t.name]
        if isinstance(grad, tf.IndexedSlices):
          grad = tf.convert_to_tensor(grad)
        variable_grads[indx] = (variable_grads[indx] + grad
                        if variable_grads[indx] is not None else grad)

    grad_x0, grad_x1 = grad_ys
    # _print_tensors(grad_x0, grad_x1, variable_grads)
    return [grad_x0, grad_x1], variable_grads

  def forward(self, x0, x1):
    custom_grad_fn = (self._efficient_grad_fn if self._use_efficient_backprop
                      else None)

    # @common_layers.fn_with_custom_grad(custom_grad_fn)
    def _forward(x0, x1):
      return _rev_mp_block_forward(x0, x1, self.f, self.g,
                                   self.num_layers)

    return _forward(x0, x1)

  def backward(self, y0, y1):
    f = list(self.f)
    g = list(self.g)
    f.reverse()
    g.reverse()
    f_vars = [v for v in tf.trainable_variables() if 'rev_mp/layer_0' in v.name]

    print (y0, y1)
    for i in range(self.num_layers):
      x0_i = y0
      if FLAGS.use_sigmoid:
        if FLAGS.function_type == "sigmoid":
          x1_i = tf.log(y1 + 1e-10) - tf.log(1 - y1 + 1e-10) - g[i](y0)
        else:
          x1_i = 0.5*(tf.log(1 + y1 + 1e-10) - tf.log(1 - y1 + 1e-10)) - g[i](x0_i)
      else:
        x1_i = y1 - g[i](x0_i)
      x1  = x1_i
      if FLAGS.use_sigmoid:
        if FLAGS.function_type == "sigmoid":
          x0 = tf.log(x0_i + 1e-10) - tf.log(1 - x0_i + 1e-10) - f[i](x1)
        else:
          x0 = 0.5*(tf.log(1 + x0_i + 1e-10) - tf.log(1 - x0_i + 1e-10)) - f[i](x1)
      else:
        x0 = x0_i - f[i](x1)
      y1, y0 = x1, x0
      # y1 = tf.Print(y1, f_vars, summarize=10, message='var')

    f.reverse()
    g.reverse()
    return y0, y1


def rev_mp_block(x0, x1, f, g, num_layers=1, is_training=True):
  """Block of reversible message passing."""
  print ('INFO: Number of layers for message passing is ', num_layers)
  tf.logging.debug('Number of message passing layers: %d', num_layers)
  rev_mp_unit = RevMessagePassingBlock(f, g, num_layers, is_training)
  return rev_mp_unit.forward(x0, x1)

def rev_mp_block_backward(y0, y1, f, g, num_layers=1, is_training=True):
  """Block of reversible message passing."""
  print ('INFO: Number of layers for message passing is ', num_layers)
  tf.logging.debug('Number of message passing layers: %d', num_layers)
  rev_mp_unit = RevMessagePassingBlock(f, g, num_layers, is_training)
  return rev_mp_unit.backward(y0, y1)

