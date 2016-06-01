import random
import numpy as np

expand = lambda s_t: np.expand_dims(s_t, 0)

class A3C_FF(object):
  def __init__(self, worker_id, local_network, local_env, config):
    self.env = local_env
    self.network = local_network

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

  def predict(self, s_t, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - self.t) / self.ep_end_t))

    if random.random() < 0:
      action = random.randrange(self.env.action_size)
    else:
      action = self.network.predict(expand(s_t))

    return action

  def observe(self, s_t, r_t, terminal):
    r_t = max(self.min_reward, min(self.max_reward, r_t))

    if (terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
      pass
