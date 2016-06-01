import random
import numpy as np

expand = lambda s_t: np.expand_dims(s_t, 0)

class A3C_FF(object):
  def __init__(self, worker_id, sess, local_networks, local_env, apply_gradeint, config):
    self.sess = sess
    self.env = local_env
    self.networks = local_networks
    self.apply_gradeint = apply_gradeint

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

    self.prev_log_policy_sampled = {}
    self.prev_policy_entropy = {}
    self.prev_value = {}
    self.prev_reward = {}

  def predict(self, s_t, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - self.t) / self.ep_end_t))

    if random.random() < 0:
      action = random.randrange(self.env.action_size)
    else:
      action = self.networks[0].predict([s_t])
      action = action[0]

    self.prev_a[self.t] = action

    return action

  def observe(self, s_t, r_t, terminal):
    r_t = max(self.min_reward, min(self.max_reward, r_t))

    self.prev_r[self.t] = r_t
    self.prev_s[self.t] = s_t

    self.prev_value[self.t] = r_t

    if (terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
      r = {}

      if terminal:
        r[self.t] = 0.
      else:
        r[self.t] = self.networks[0].calc_value([s_t])[0][0]

      for t in xrange(self.t - 1, self.t_start - 1, -1):
        r[t] = self.prev_r[t + 1] + self.gamma * r[t + 1]

      data = {}
      data.update({
        self.networks[t].R: [r[t + self.t_start]] for t in xrange(len(self.prev_r) - 1)})
      data.update({
        self.networks[t].s_t: [self.prev_s[t + self.t_start]] for t in xrange(len(self.prev_r) - 1)})
      data.update({
        self.networks[t].true_action : [self.prev_a[t + self.t_start]] for t in xrange(len(self.prev_r) - 1)})

      self.sess.run(self.apply_gradeint[len(self.prev_r) - 1], feed_dict=data)

      #self.copy_from_global()

      self.prev_a = {self.t: self.prev_a[self.t]}
      self.prev_s = {self.t: self.prev_s[self.t]}
      self.prev_r = {self.t: self.prev_r[self.t]}
      self.t_start = self.t

    self.t += 1
