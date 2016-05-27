import logging
import numpy as np
import tensorflow as tf

from .deep_q_network import DeepQNetwork
from models.environment import Environment
from .utils import accumulate_gradients

logger = logging.getLogger(__name__)

class A3C_FF(object):
  def __init__(self, thread_id, config, sess, global_model, global_optim):
    self.sess = sess
    self.thread_id = thread_id
    self.global_model = global_model
    self.global_optim = global_optim

    self.env = Environment(config.env_name,
                           config.n_action_repeat,
                           config.max_random_start,
                           config.history_length,
                           config.screen_height,
                           config.screen_width)

    self.t = 0
    self.t_start = 0
    self.t_max = config.t_max

    self.max_reward = config.max_reward
    self.min_reward = config.min_reward

    self.screen_height = config.screen_height
    self.screen_width = config.screen_width
    self.data_format = config.data_format
    self.history_length = config.history_length

    #self.prev_s = np.empty((self.t_max, self.history_length, config.screen_height, config.screen_width), dtype=np.float16)
    self.prev_p_logits = np.empty(self.t_max, dtype=np.integer)
    self.prev_r = np.empty(self.t_max, dtype=np.integer)
    self.prev_v = np.empty(self.t_max, dtype=np.integer)
    self.prev_t = np.empty(self.t_max, dtype=np.bool)

    self.build_model()

  def build_model(self):
    self.networks, grads = [], []

    with tf.variable_scope('thread%d' % self.thread_id) as scope:
      self.optim_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

      for step in xrange(self.t_max):
        with tf.name_scope('A3C_%d' % step) as scope:
          network = DeepQNetwork(self.sess, self.data_format,
                                self.history_length,
                                self.screen_height,
                                self.screen_width,
                                self.env.action_size)
          self.networks.append(network)

          tf.get_variable_scope().reuse_variables()
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          grad = self.global_optim.compute_gradients(network.total_loss)
          grads.append(grad)

      # Accumulate gradients for n-steps
      accumulated_grads = accumulate_gradients(grads)

      for grad, var in accumulated_grads:
        if grad is not None:
          summaries.append(
              tf.histogram_summary('%s/gradients' % var.op.name, grad))

      self.apply_gradeint_op = self.global_optim.apply_gradients(accumulated_grads, global_step=self.optim_step)

  def act(self, s_t, r_t, terminal):
    logger.info(" [%d] ACT : %s, %s" % (self.thread_id, r_t, terminal))

    # clip reward
    if self.max_reward:
      r_t = min(self.max_reward, r_t)
    if self.min_reward:
      r_t = max(self.min_reward, r_t)

    self.prev_r[self.t - 1] = r_t

    if (terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
      r, a, s = {}, {}, {self.t: s_t}

      if terminal:
        r[self.t] = 0
      else:
        r[self.t] = self.networks[0].calc_value([s_t])

      for t in xrange(self.t - 1, self.t_start, -1):
        r[t] = self.prev_r[t + 1] + self.gamma * r[t]

      data = {}
      data.update({network.R: r[t] for t, network in enumerate(self.networks)})
      data.update({network.policy_logits: self.prev_p_logits[t] for t, network in enumerate(self.networks)})
      data.update({network.value: self.prev_v[t] for t, network in enumerate(self.networks)})

      self.sess.run(self.apply_gradeint_op, feed_dict=data)
      self.copy_from_global()

      self.prev_r *= 0

      self.t_start = self.t
      action = None
    else:
      p_logits, v, action = self.networks[0].calc_policy_logits_value_action([s_t])
      action = action[0]

      self.prev_p_logits[self.t] = p_logits[0][action]
      self.prev_v[self.t] = v[0][0]

      self.t += 1

    return action

  def copy_from_global(self):
    for network in self.networks:
      network.copy_w_from(self.global_model)
