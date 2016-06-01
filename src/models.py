import re
import random
import numpy as np
import tensorflow as tf

from .utils import range

expand = lambda s_t: np.expand_dims(s_t, 0)

class A3C_FF(object):
  def __init__(self, worker_id, sess, local_networks, local_env,
               global_network, global_optim, config):
    self.sess = sess

    self.env = local_env
    self.networks = local_networks

    self.global_network = global_network
    self.global_optim = global_optim

    self.t = 0
    self.t_start = 0
    self.t_max = config.t_max

    self.ep_start = config.ep_start
    self.ep_end = config.ep_end
    self.ep_end_t = config.ep_end_t

    self.gamma = config.gamma
    self.max_reward = config.max_reward
    self.min_reward = config.min_reward

    self.data_format = config.data_format
    self.action_size = self.env.action_size
    self.screen_height = config.screen_height
    self.screen_width = config.screen_width
    self.history_length = config.history_length

    self.prev_p_logits = {} # np.empty([self.t_max, self.action_size], dtype=np.integer)
    self.prev_s = {} # np.empty([self.t_max, 1], dtype=np.integer)
    self.prev_v = {} # np.empty([self.t_max, 1], dtype=np.integer)
    self.prev_r = {} # np.empty(self.t_max, dtype=np.integer)
    self.prev_log_policy = {} # np.empty(self.t_max, dtype=np.integer)
    self.prev_t = {} # np.empty(self.t_max, dtype=np.bool)

    self.prev_log_policy_sampled = {}
    self.prev_policy_entropy = {}
    self.prev_value = {}
    self.prev_reward = {}

    self.s_t_shape = self.networks[0].s_t.get_shape().as_list()
    self.s_t_shape[0] = 1

    self.make_accumulated_gradients()

  # Add accumulated gradient ops for n-step Q-learning
  def make_accumulated_gradients(self):
    global_var = {w.name.replace('A3C_global/', ''):w for w in self.global_network.w.values()}

    reset_accum_grads = []
    new_grads_and_vars = []

    # 1. accum_grads
    self.accum_grads = {}
    self.add_accum_grads = {}

    for step, network in enumerate(self.networks):
      grads_and_vars = self.global_optim.compute_gradients(network.total_loss, network.w.values())
      _add_accum_grads = []

      for grad, var in tuple(grads_and_vars):
        if grad is not None:
          shape = grad.get_shape().as_list()

          name = 'accum/%s' % "/".join(var.name.split(':')[0].split('/')[-3:])
          if step == 0:
            self.accum_grads[name] = tf.Variable(
                tf.zeros(shape), trainable=False, name=name)

            global_v = global_var[re.sub(r'.*\/A3C_\d+\/', '', var.name)]
            new_grads_and_vars.append((self.accum_grads[name].ref(), global_v))

            reset_accum_grads.append(self.accum_grads[name].assign(tf.zeros(shape)))

          _add_accum_grads.append(tf.assign_add(self.accum_grads[name], grad))

      # 2. Add gradient to accum_grads
      self.add_accum_grads[step] = tf.group(*_add_accum_grads)

    # 3. Reset accum_grads
    self.reset_accum_grad = tf.group(*reset_accum_grads)

    # 4. Update variables of global_network with accum_grads
    self.apply_gradient = self.global_optim.apply_gradients(new_grads_and_vars)

    for step, add_accum_grads in self.add_accum_grads.items():
      with tf.control_dependencies([add_accum_grads]):
        self.add_accum_grads[step] = tf.constant(0)

    # Add dummy_op to execute optimizer with partial_run
    with tf.control_dependencies([self.apply_gradient]):
      self.fake_apply_gradient = tf.constant(0)

  def reset_partial_graph(self):
    targets = [network.sampled_action for network in self.networks]
    targets.extend([network.log_policy_of_sampled_action for network in self.networks])
    targets.extend([network.value for network in self.networks])
    targets.extend(self.add_accum_grads.values())
    targets.append(self.fake_apply_gradient)

    inputs = [network.s_t for network in self.networks]
    inputs.extend([network.R for network in self.networks])
    inputs.extend([network.true_log_policy for network in self.networks])

    self.partial_graph = self.sess.partial_run_setup(targets, inputs)

  def predict(self, s_t, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - self.t) / self.ep_end_t))

    if self.t_start - self.t == 0:
      self.reset_partial_graph()

    if random.random() < 0:
      action = random.randrange(self.env.action_size)
    else:
      network_idx = self.t - self.t_start
      action, log_policy = self.sess.partial_run(
          self.partial_graph,
          [
            self.networks[network_idx].sampled_action,
            self.networks[network_idx].log_policy_of_sampled_action,
          ],
          {
            self.networks[network_idx].s_t: [s_t]
          }
      )
      action, log_policy = action[0], log_policy[0]

    self.prev_log_policy[self.t] = action
    self.t += 1

    return action

  def observe(self, s_t, r_t, terminal):
    r_t = max(self.min_reward, min(self.max_reward, r_t))

    self.prev_r[self.t] = r_t
    self.prev_s[self.t] = s_t

    self.prev_value[self.t] = r_t

    if (terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
      print terminal
      r = {}

      if terminal:
        r[self.t] = 0.
      else:
        r[self.t] = self.sess.partial_run(
            self.partial_graph,
            self.networks[self.t_start - self.t].value,
        )[0][0]

      for t in range(self.t - 1, self.t_start - 1, -1):
        r[t] = self.prev_r[t] + self.gamma * r[t + 1]

      data = {}
      data.update({
        self.networks[t].R: [r[t + self.t_start]] for t in range(len(self.prev_r) - 1)
      })
      data.update({
        self.networks[t].true_log_policy:
          [self.prev_log_policy[t + self.t_start]] for t in range(len(self.prev_r) - 1)
      })

      self.sess.partial_run(self.partial_graph,
          [self.add_accum_grads[t] for t in range(len(self.prev_r) - 1)], data)

      # Reset accumulated gradients to zero
      self.sess.run(self.reset_accum_grad)

      # Copy w of global_network tot local_network
      #self.copy_from_global()

      self.prev_s = {self.t: self.prev_s[self.t]}
      self.prev_r = {self.t: self.prev_r[self.t]}

      self.t_start = self.t
