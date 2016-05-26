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
    self.n_step = config.n_step

    self.max_reward = config.max_reward
    self.min_reward = config.min_reward

    self.data_format = config.data_format
    self.history_length = config.history_length
    self.screen_height = config.screen_height
    self.screen_width = config.screen_width

    self.s_ts = np.empty((self.n_step, config.screen_height, config.screen_width), dtype = np.float16)
    self.prev_rewards = np.empty(self.n_step, dtype = np.integer)
    self.prev_values = np.empty(self.n_step, dtype = np.integer)

    self.build_model()

  def build_model(self):
    self.networks, grads = [], []

    with tf.variable_scope('t%d' % self.thread_id) as scope:
      for step in xrange(self.n_step):
        with tf.variable_scope('A3C_%d' % step) as scope:
          network = DeepQNetwork(self.data_format,
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

      self.optim_step = tf.placeholder('int32', None, name='optim_step')

      import ipdb; ipdb.set_trace() 
      self.apply_gradeint_op = self.global_optim.apply_gradients(
          self.global_model.variables, accumulated_grads)

      if False: #self.pid == 0:
        for var in tf.trainable_variables():
          summaries.append(tf.histogram_summary(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        self.train_op = tf.group(
            self.policy_apply_gradeint_op, self.value_apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

  def act(self, s_t, reward, terminal):
    logger.info(" [%d] ACT : %s, %s" % (self.thread_id, reward, terminal))

    # clip reward
    if self.max_reward:
      reward = min(self.max_reward, reward)
    if self.min_reward:
      reward = max(self.min_reward, reward)

    self.prev_rewards[self.t - 1] = reward

    if (terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
      r, a, s = {}, {}, {t: s_t}
      if terminal:
        r[t] = 0
      else:
        _, r[t] = self.networks[0].calc_policy_value(s_t)

      for t in xrange(self.t_start, self.t, -1):
        r[t] = self.prev_r[t] + self.gamma * r[t]

      data = {}
      data.update({network.s_t: self.prev_s[t] for network in self.networks})
      data.update({network.R: r[t] for network in self.networks})

      self.sess.run(self.self.train_op, feed_dict=data)
    else:
      Q = self.sess.run([self.networks[0].value], {self.networks[0].s_t: s_t})

    if not terminal:
      self.prev_s[self.t] = s_t
      self.prev_r[self.t] = reward
      self.prev_t[self.t] = terminal
      self.t += 1
    else:
      print "Should implement"
      pass

  def copy_from_global(self):
    for network in self.networks:
      network.copy_w_from(self.global_model)
