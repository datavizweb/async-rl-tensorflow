import random

class A3C_FF(object):
  def __init__(self, worker_id, global_network, global_optim, local_network, local_env, config):
    self.env = local_env
    self.network = local_network
    self.global_optim = global_optim
    self.global_network = global_network

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

  def predict(self, state, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - self.t) / self.ep_end_t))

    if random.random() < 0:
      action = random.randrange(self.env.action_size)
    else:
      action = self.network.pred_action.eval({self.network.s_t: [state]})

    return action
