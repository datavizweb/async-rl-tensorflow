import cv2
import gym
import random
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Environment(object):
  def __init__(self, env_name, n_action_repeat, max_random_start,
               history_length, screen_height=84, screen_width=84):
    self.env = gym.make(env_name)

    self.n_action_repeat = n_action_repeat
    self.max_random_start = max_random_start

    self.history_length = history_length
    self.action_size = self.env.action_space.n

    self.screen_width = screen_width
    self.screen_height = screen_height

    self.history = np.zeros(
        [history_length, self.screen_height, self.screen_width], dtype=np.uint8)

    logger.info("Using %d actions : %s" % (self.action_size, ", ".join(self.env.get_action_meanings())))

  def new_game(self, from_random_game=False):
    self.history *= 0

    screen = self.env.reset()
    screen, reward, terminal, _ = self.env.step(0)

    if from_random_game == False:
      self._add_history(screen)
      self.lives = self.env.ale.lives()

    # history, reward, terminal
    return self.history, 0, False

  def new_random_game(self):
    screen, reward, terminal = self.new_game(True)

    for idx in xrange(random.randrange(self.max_random_start)):
      screen, reward, terminal, _ = self.env.step(0)

      if terminal: logger.warning("WARNING: Terminal signal received after %d 0-steps", idx)

    self._add_history(screen)
    self.lives = self.env.ale.lives()

    # history, reward, terminal
    return self.history, 0, False

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
      self._add_history(screen)
      self.lives = current_lives

    return self.history, reward, terminal

  def _add_history(self, raw_screen):
    y = 0.2126 * raw_screen[:, :, 0] + 0.7152 * raw_screen[:, :, 1] + 0.0722 * raw_screen[:, :, 2]
    y = y.astype(np.uint8)
    y_screen = cv2.resize(y, (self.screen_height, self.screen_width))

    self.history[:-1] = self.history[1:]
    self.history[-1] = y_screen
