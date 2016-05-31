class A3C_FF(object):
  def __init__(self, worker_id, global_network, global_optim, local_network, local_env):
    self.network = local_network
    self.env = local_env

  def predict(self, state):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - self.t) / self.ep_end_t))

    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.network.action(state)
    return action
