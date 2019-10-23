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

"""GNN which implements Neumann RBP as the backpropagation method."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.layers import common_layers
import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_float('eps', 1e-6, 'Epsilon to guarantee convergence.')
FLAGS = flags.FLAGS

def _get_variable_util(f, variables):
  """Utility function to return the variables in the function argument."""
  f_vars = []
  for i, var in enumerate(variables):
    print (var.name)
    if 'neumann_mp/f' in var.name:
      f_vars.append(var)
  return f_vars

def base_forward(x, f, max_num_steps, is_trainable=False):
  """Define the forward pass for RBP (and Neumann RBP).

  Run the forward pass for base RBP, where the number of time steps
  to run the algorithm is limited by the max_num_steps. This is needed
  as we have to compute dh*/dh.

  Args:
    x: A tensor which represents the state which is updated during propagation.
    f: A function which is recursively applied at each step of propagation.
    max_num_steps: Number of steps to run RBP for. Usually in the order of 100.
      Best performance found at 100 steps.
    is_trainable: A boolean flag to decide if vars in f are trainable or not.

  Return:
    x_prev: A tensor containing the value of the state at time t-1
    x_curr: A tensor containing the value of the state at time t (convergence)
    num_steps: A tensor containing the number of steps for which the
      RBP was run.
  """
  def _condition(x_prev, x_curr, num_steps):
    """Return True or False depending upon whether condition holds or not."""
    diff = (tf.reduce_sum(tf.square(x_prev - x_curr)) > FLAGS.eps)
    diff = tf.logical_or(diff, tf.equal(num_steps, 0))
    return tf.logical_and(diff, num_steps < max_num_steps)

  def step_grad_fn(inputs, variables, ys, grad_ys):
    """Define custom gradients through the while loop during RBP.

    Regular gradient propagation through a tf.while_loop would be of the form
    of TBPTT (Truncated Backprop through time). This also accumulates gradients
    of the loss with respect to the variables. In RBP, the variables are just
    updates based on the final converged hidden state value h*, and hence TBPTT
    must be disabled. This custom gradient op disables TBPTT, by returning
    None gradients for the variables as well as just passing the 'stateless'
    dL/dh* back to the initial value of the hidden state before propagation,
    so that everything before that has a gradient signal.
    """
    _, grad_x_tp2, _ = grad_ys
    x_tp1, x_tp2, _ = ys
    x_t = inputs[0]
    x_tp1_1 = inputs[1]
    x_tp1_stop = tf.stop_gradient(x_tp1)
    grad_x_tp1 = tf.gradients(f(x_tp1_stop), x_tp1_stop, grad_x_tp2)[0]
    return [None, grad_x_tp1, None], [None]*len(variables)

  @common_layers.fn_with_custom_grad(step_grad_fn)
  def _body(x_prev, x_curr, num_steps):
    """Perform the body of the while loop implementing RBP.

    Args:
      x_prev: A tensor containing the value of the state at time num_steps-1
      x_curr: A tensor containing the value of the state at time num_steps
      num_steps: A tensor containing the number of steps for which the
        RBP was run till now
    """
    x_t = f(x_curr, is_trainable=is_trainable)
    x_prev = x_curr
    x_curr = x_t
    num_steps = num_steps + 1
    return x_prev, x_curr, num_steps

  x_prev, x_curr = x, x
  num_steps = tf.zeros(shape=(), dtype=tf.int32)
  # Define the while loop with custom gradient for the forward propagation
  # step function which implements the differed gradient flow as in RBP.
  x_prev, x_curr, num_steps = tf.while_loop(_condition,
                                            _body,
                                            loop_vars=[x_prev, x_curr, num_steps],
                                            back_prop=True)
  return x_prev, x_curr, num_steps

def _neumann_forward(x, f, is_trainable=True):
  """Run one step of forward propagation of Neumann RBP algorithm."""
  x_new = f(x, is_trainable=is_trainable)
  return x, x_new

class NeumannMessagePassingBlock(object):
  """Class to define and run neumann RBP for variable number of steps.

  This class implements a block which runs variable number of steps of
  message passing until convergence. Message passing function is provided by
  the model and the backprop (RBP) is done once converged to the final value.

  Attributes:
    f: A function which specifies the message passing.
    num_steps: Maximum number of steps to run for searching for convergence.
    _use_neumann_backprop: A boolean flag to indicate whether Neumann RBP
      is to be used or not.
    rbp_steps: Number of terms in the Neumann expansion to be used to
      compute gradient. (This is different from the number of steps to run
      message passing for.)
  """

  def __init__(self, f,
               max_num_steps,
               is_training=True,
               use_neumann_backprop=True,
               rbp_steps=50):
    scope_prefix = "neumann_mp/f"
    self.f = f
    self.num_steps = max_num_steps
    self.max_num_steps = max_num_steps
    self._use_neumann_backprop = use_neumann_backprop
    self.rbp_steps = rbp_steps

  def forward(self, x):
    """Forward propagation for the purpose of Neumann Message Passing."""
    custom_grad_fn = (self.neumann_rbp if self._use_neumann_backprop
                      else None)
    @common_layers.fn_with_custom_grad(custom_grad_fn)
    def _forward(x):
      out =  _neumann_forward(x, self.f, self.max_num_steps)
      return out[1]

    def _base_forward(x):
      out = base_forward(x, self.f, self.max_num_steps,
                         is_trainable=False)
      return out

    # Base forward is the function which immplements Neumann RBP.
    # First run the regular recursive propagation untill convergence and then
    # define the RBP gradient on the final values (after convergence) by once
    # again applying the forward propagation but this time it is with the RBP
    # custom gradient. _base_forward(x) just makes sure that TBPTT is not
    # implemented (which is the default backprop through the while loop).
    x_prev, x_curr, num_steps_run = _base_forward(x)
    return _forward(x_curr)


  def neumann_rbp(self, inputs,
                  variables,
                  ys,
                  grad_ys):
    """Implement Neumann RBP on the converged value of hidden state.

    Args:
      inputs: A tensor which represents input states over which RBP is applied.
      variables: A list of tf.variables through which gradient has to flow.
      ys: The output of the message passing propagation procedure.
      grad_ys: The gradient of the loss with respect to the output of MP.

    Return:
      grad_h_prev: The gradient of the loss with respect to the hidden state
        at the end of propagation (ideally, the converged value)
      grads_w_f: The gradient of the loss with respect to the variables. In RBP
        this is only determined by the final converged state.
    """
    f_vars = _get_variable_util(self.f, variables)
    f_var_grads = []

    # ys = [x_prev, x_curr] as this is the output of the forward function
    # grad_ys = complete gradients of x_prev and x_curr
    # We need to find the gradient of L w.r.t. x which is the input to the
    # function, and also find the gradient of the loss with respect to the
    # variables in function f. The latter is done through neumann RBP, which
    # only involves the final converged values of x = x* = x_curr = x_prev.
    # The former is done through the regular chain in the stochastic
    # computation graph and for this the while loop should be backprop'able.
    grad_h_curr = grad_ys
    h_curr = ys
    h_prev = inputs[0]
    h_prev_stop = tf.stop_gradient(h_prev)
    fh_prev = self.f(h_prev_stop, is_trainable=True)
    neumann_v = grad_h_curr[0]
    neumann_g = grad_h_curr[0]
    print (neumann_v, neumann_g)

    for i in range(self.rbp_steps):
      neumann_v = tf.gradients(fh_prev, h_prev_stop,
                               grad_ys=neumann_v)[0]
      neumann_g += neumann_v

    # neumann_v iteratively just computes the power sum of J_{F, h*}'
    # neumann_g accumulates the sum in itself. Now, grads_w_f would be just
    # df(x, h*)/dw * neumann_g.
    grads_w_f = tf.gradients(fh_prev, f_vars, neumann_g)

    # Now accumulate gradients for getting gradients with respect to the
    # inputs. That is, if the iterative sequence is h_0, h_1, ..., h*, we want
    # to compute the gradient upto dL/dh*.
    grad_h_prev = tf.gradients(fh_prev, h_prev_stop, grad_ys=grad_h_curr)[0]
    return [grad_h_prev], grads_w_f

def neumann_mp_block(x, f, max_num_steps=50, rbp_steps=10, is_training=True):
  """Wrapper over the forward function implementing Neumann RBP."""
  print ('Initializing Neumann Message Passing Block.')
  neumann_mp_unit = NeumannMessagePassingBlock(f, max_num_steps, is_training,
                                                 rbp_steps=rbp_steps)
  return neumann_mp_unit.forward(x)
