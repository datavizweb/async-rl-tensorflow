import gym
import random

class Environment(object):
  def __init__(self, env_id,
               history_length, screen_height, screen_width,
               action_repeat, max_random_start, use_cpu=True):
    self.env = gym.make(env_id)

    self.action_repeat = action_repeat
    self.max_random_start = max_random_start
    self.action_size = self.env.action_space.n

    if use_cpu:
      self.history = np.array([history_length, screen_height, screen_width])
    else:
      self.history = np.array([history_length, screen_height, screen_width])

  def new_game(self):
    self.env.reset()

    for _ in xrange(random.randrange(max_random_start)):
      state, reward, terminal = self.env.step(0)

    state = self.update_history(screen)
    return state, reward, terminal

  def step(self, action):
    cumulative_reward = 0

    for _ in xrange(action_repeat):
      screen, reward, terminal = self.env.step(action)
      cumulative_reward += reward
      
      if terminal: break

    state = self.update_history(screen)
    return state, cumulative_reward, terminal

  def update_history(self, screen):
    self.history[:-1,:,:] = self.history[1:,:,:]
    self.history[-1,:,:] = screen
    return self.history
