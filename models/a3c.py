import random
import logging
import numpy as np
import tensorflow as tf

from .deep_q_network import DeepQNetwork
from models.environment import Environment
from .utils import accumulate_gradients, save_history_as_image

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
                           config.data_format,
                           config.display,
                           config.screen_height,
                           config.screen_width)

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
    self.prev_a = {} # np.empty(self.t_max, dtype=np.integer)
    self.prev_t = {} # np.empty(self.t_max, dtype=np.bool)

    self.build_model()

  def build_model(self):
    self.networks, grads = [], []


    with tf.variable_scope('thread%d' % self.thread_id) as scope:
      self.optim_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

      self.contain_grads = []
      for step in xrange(self.t_max):
        with tf.name_scope('A3C_%d' % step) as scope:
          network = DeepQNetwork(self.sess, self.data_format,
                                self.history_length,
                                self.screen_height,
                                self.screen_width,
                                self.action_size)
          self.networks.append(network)

          tf.get_variable_scope().reuse_variables()
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          grad = self.global_optim.compute_gradients(network.total_loss)
          grads.append(grad)

      # Accumulate gradients for n-steps
      self.apply_gradeint_op = {}
      for step in xrange(1, self.t_max + 1):
        accumulated_grads = accumulate_gradients(grads[:step])

        self.apply_gradeint_op[step] = \
            self.global_optim.apply_gradients(accumulated_grads, global_step=self.optim_step)

  def predict(self, s_t, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - self.t) / self.ep_end_t))

    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      p_logits, v, action = self.networks[0].calc_policy_logits_value_action([s_t])
      action = action[0]

    #self.prev_p_logits[self.t] = p_logits[0]
    #self.prev_v[self.t] = v[0]
    self.prev_a[self.t] = action

    return action

  def observe(self, s_t, r_t, terminal):
    logger.info("%2d [%6d] r: %s, t: %s" % (self.thread_id, self.t, r_t, terminal))

    r_t = max(self.min_reward, min(self.max_reward, r_t))
    self.prev_r[self.t] = r_t
    self.prev_s[self.t] = s_t.copy()

    if (terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
      r = {}

      if terminal:
        r[self.t] = 0.
      else:
        r[self.t] = self.networks[0].calc_value([s_t])[0][0]

      for t in xrange(self.t - 1, self.t_start - 1, -1):
        r[t] = self.prev_r[t + 1] + self.gamma * r[t + 1]

      print r
      print self.prev_r

      data = {}
      data.update({
        self.networks[t].R: [r[t + self.t_start]] for t in xrange(len(self.prev_r) - 1)})
      data.update({
        self.networks[t].s_t: [self.prev_s[t + self.t_start]] for t in xrange(len(self.prev_r) - 1)})
      data.update({
        self.networks[t].true_action : [self.prev_a[t + self.t_start]] for t in xrange(len(self.prev_r) - 1)})
      #data.update({network.policy_logits: [self.prev_p_logits[t]] for t, network in enumerate(self.networks)})
      #data.update({network.value: [self.prev_v[t]] for t, network in enumerate(self.networks)})
      #data.update({network.value: [self.prev_v[t]] for t, network in enumerate(self.networks)})

      #  for i in xrange(4):
      #    print np.sum(self.prev_s[t + self.t_start][:,:,i])

      self.sess.run(self.apply_gradeint_op[len(self.prev_r) - 1], feed_dict=data)
      #_, p0, p1, p2, p3, e0, e1, e2, e3 = self.sess.run([
      #  self.apply_gradeint_op[len(self.prev_r) - 2],
      #  self.networks[0].policy,
      #  self.networks[1].policy,
      #  self.networks[2].policy,
      #  self.networks[3].policy,
      #  self.networks[0].entropy,
      #  self.networks[1].entropy,
      #  self.networks[2].entropy,
      #  self.networks[3].entropy,
      #  ], feed_dict=data)

      #logger.info("entropy: %s, probs: %s" % (e0, p0))
      #logger.info("entropy: %s, probs: %s" % (e1, p1))
      #logger.info("entropy: %s, probs: %s" % (e2, p2))
      #logger.info("entropy: %s, probs: %s" % (e3, p3))

      #import ipdb; ipdb.set_trace() 
      self.copy_from_global()

      self.prev_a = {self.t: self.prev_a[self.t]} # *= 0
      self.prev_s = {self.t: self.prev_s[self.t]} # *= 0
      self.prev_r = {self.t: self.prev_r[self.t]} # *= 0
      self.t_start = self.t

    self.t += 1

  def copy_from_global(self):
    for network in self.networks:
      network.copy_w_from(self.global_model)
