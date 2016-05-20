import gym

class Environment(object):
  def __init__(self, env_name, n_frame_skip, n_action_repeat,
               dead_as_terminal, max_random_start,
               screen_width=84, screen_height=84):
    self.env = gym.make(env_name)

    self.n_frame_skip = n_frame_skip
    self.n_action_repeat = n_action_repeat
    self.dead_as_terminal = dead_as_terminal
    self.max_random_start = max_random_start
    self.action_size = self.env.action_space.n

    self.screen_width = screen_width
    self.screen_height = screen_height

    self.history = np.zeros(
        [history_length, self.screen_height, self.screen_width], dtype=np.uint8)

  def new_game(self, from_random_game=False):
    self.history *= 0

    screen = self.env.reset()
    screen, reward, terminal, _ = self.env.step(0)

    if from_random_game == False:
      self.add_history(screen)
      self.lives = self.env.lives()

    return self.history, reward, terminal

  def new_random_game(self):
    screen, reward, terminal = self.new_game(True)

    for _ in xrange(random.randrange(self.self.max_random_start)):
      screen, reward, terminal, _ = self.env.step(0)

    self.add_history(screen)
    self.lives = self.env.lives()

    return self.history, reward, terminal 

  def step(self, action, is_training):
    cumulated_reward = 0

    for _ in xrange(self.n_action_repeat):
      screen, reward, terminal, _ = self.env.step(action)
      cumulated_reward += reward
      current_lives = self.env.lives()

      if is_training and self.lives > current_lives:
        terminal = True

      if terminal: break

    if not terminal:
      self.add_history(screen)
      self.lives = current_lives

    return self.history, reward, terminal

  def _add_history(self):
    self.history[:-1] = self.history[1:]
    self.history[-1] = self._screen

  @property
  def screen(self):
    # Luminance
    y = 0.2126 * self._screen[:, :, 0] + 0.7152 * self._screen[:, :, 1] + 0.0722 * self._screen[:, :, 2]
    y = y.astype(np.uint8)
    return cv2.resize(y, (self.screen_height, self.screen_width))

  @property
  def state(self):
    return self._screen, self.reward, self.terminal
