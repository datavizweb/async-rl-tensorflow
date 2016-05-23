import copy
import multiprocessing as mp

class Agent(object):
  def __init__(self):
    t = mp.Value('t', 0)

    self.policy_optim = tf.train.RMSPropOptimizer(
        self.learning_rate, self.decay, self.momentum, self.epsilon, name='policy_RMSProp')
    self.value_optim = tf.train.RMSPropOptimizer(
        self.learning_rate, self.decay, self.momentum, self.epsilon, name='value_RMSProp')

  def make_env(self):
    return Environment(config.env_name, config.n_action_repeat, config.max_random_start,
        config.history_length, config.screen_height, config.screen_width)

  def build_model(self):
    def make_model():
      model = A3C_FF()
      optim = tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon).minimize(model.cost)

      return model, optim

    env = self.make_env()

    def func(env, process_id):
      model, optim = make_model()

      dqn = NatureDQN(data_format, config.history_length,
          config.screen_height, config.screen_width, env.action_size, name='nature')

  def train(self):
    train_threads = []
    for i in range(self.parallel_size):
      train_threads.append(threading.Thread(target=train_function, args=(i,)))
      
    while:
      with counter.get_lock():
        counter.value += 1
        global_t = counter.value

      if global_t > step:
        break

      action = agent.act(env.

  def evaluation(self, func, n_runs):
    rewards = []
    for i in xrange(n_runs):
      current_reward = 0
      env = self.make_env()

      state, reward, terminal = env.new_random_game()
      while not terminal:
        state, reward, terminal = env.step(action)
        current_reward += reward

      rewards.append(current_reward)

    return rewards
