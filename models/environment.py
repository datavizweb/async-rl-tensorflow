import gym

class State(object):
  def __init__(self):
    self.reward = None
    self.terminal = None
    self.prev_lives = None
    self.lives = None

class Environment(object):
  def __init__(self, env_name, n_frame_skip,
               dead_as_terminal, max_random_start):
    self.env = gym.make(env_name)

    self.n_frame_skip = frame_skip
    self.dead_as_terminal = dead_as_terminal
    self.max_random_start = max_random_start
    self.action_size = self.env.action_space.n

  def new_game(self, from_random_game=False):
    self._observation = self.env.reset()
    self._step(0)
    self.render()
    return self.observation, 0, 0, self.terminal

  def _step(self, action):
    return self.env.step(action)

  def _update_state(self, observation, reward, terminal, lives):
    self._state.

  def new_random_game(self):
    self.new_game(True)
    for _ in xrange(random.randint(0, self.random_start - 1)):
      self._step(0)
    self.render()
    return self.observation, 0, 0, self.terminal 

  def step(self):
    pass

  @property
  def state(self):
    pass

  @property
  def reward(self):
    pass
