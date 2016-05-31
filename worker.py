class Worker(object):
  def __init__(self, worker_id, global_network, global_optim, local_network, local_env):
    self.network = local_network
    self.env = local_env

    DeepQNetwork
