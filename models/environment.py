import gym

class Environment(object):
  def __init__(self, env_name, n_frame_skip,
               dead_as_terminal, max_random_start):
    self.env = gym.make(env_name)
    self.n_frame_skip = frame_skip
    self.dead_as_terminal = dead_as_terminal:wq
    self.max_random_start = max_random_start

  def step(self):
    pass

  @property
  def state(self):
    pass

  @property
  def reward(self):
    pass
