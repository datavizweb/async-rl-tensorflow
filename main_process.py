import gym
import time
import random
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import multiprocessing as mp

from src.utils import timeit, range
from src.models import A3C_FF
from src.network import Network
from src.environment import Environment

flags = tf.app.flags

# Deep q Network
flags.DEFINE_string('data_format', 'NHWC', 'The format of convolutional filter')
flags.DEFINE_string('ep_start', 1., 'The value of epsilon at start in e-greedy')
flags.DEFINE_string('ep_end', 0.1, 'The value of epsilnon at the end in e-greedy')
flags.DEFINE_string('ep_end_t', 1000000, 'The time t when epsilon reach ep_end')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('n_action_repeat', 4, 'The number of actions to repeat')
flags.DEFINE_integer('max_random_start', 30, 'The maximum number of NOOP actions at the beginning of an episode')
flags.DEFINE_integer('screen_height', 84, 'The height of gym screen')
flags.DEFINE_integer('screen_width', 84, 'The width of gym screen')
flags.DEFINE_integer('history_length', 4, 'The length of history of screens to use as an input to DQN')
flags.DEFINE_integer('max_reward', +1, 'The maximum value of clipped reward')
flags.DEFINE_integer('min_reward', -1, 'The minimum value of clipped reward')

# Training
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_float('learning_rate', 7e-4, 'The learning rate of training')
flags.DEFINE_float('decay', 0.99, 'Decay of RMSProp optimizer')
flags.DEFINE_float('epsilon', 0.1, 'Epsilon of RMSProp optimizer')
flags.DEFINE_float('momentum', 0.0, 'Momentum of RMSProp optimizer')
flags.DEFINE_float('gamma', 0.99, 'Discount factor of return')
flags.DEFINE_float('beta', 0.0, 'Beta of RMSProp optimizer')
flags.DEFINE_integer('t_max', 5, 'The maximum number of t while training')
flags.DEFINE_integer('n_worker', 4, 'The number of workers to run asynchronously')

# Debug
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_string('log_level', 'INFO', 'Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

config = flags.FLAGS

logger = logging.getLogger()
logger.propagate = False

logger.setLevel(config.log_level)

# set random seed
tf.set_random_seed(config.random_seed)
random.seed(config.random_seed)

def main(_):
  action_size = gym.make(config.env_name).action_space.n

  def make_network(sess, global_network=None, global_optim=None, name=None):
    with tf.variable_scope(name) as scope:
      return Network(sess, config.data_format,
              config.history_length,
              config.screen_height,
              config.screen_width,
              action_size,
              global_network=global_network,
              global_optim=global_optim)

  server = tf.train.Server.create_local_server()

  main_sess = tf.Session(server.target)
  global_network = make_network(main_sess, name='A3C_global')
  global_optim = tf.train.RMSPropOptimizer(config.learning_rate,
                                            config.decay,
                                            config.momentum,
                                            config.epsilon)

  tf.initialize_all_variables().run(session=main_sess)

  @timeit
  def worker_func(worker_id):
    with tf.Session(server.target) as sess:
      with tf.variable_scope('thread%d' % worker_id) as scope:
        networks, grads = [], []

        for step in range(config.t_max):
          network = make_network(sess, global_network, global_optim, name='A3C_%d' % (worker_id))
          networks.append(network)

          tf.get_variable_scope().reuse_variables()

      env = Environment(config.env_name, config.n_action_repeat, config.max_random_start,
                        config.history_length, config.data_format, config.display,
                        config.screen_height, config.screen_width)
      model = A3C_FF(worker_id, sess, networks, env, global_network, global_optim, config)
      tf.initialize_all_variables().run(session=sess)

      state, reward, terminal = model.env.new_random_game()
      model.observe(state, reward, terminal)

      start_time = time.time()
      for idx in range(100):
        print worker_id, idx
        # 1. predict
        action = model.predict(state)
        # 2. step
        state, reward, terminal = model.env.step(-1, is_training=True)
        # 3. observe
        model.observe(state, reward, terminal)

        if terminal:
          model.env.new_random_game()
      logger.info("loop : %2.2f sec" % (time.time() - start_time))

  # Prepare each workers to run asynchronously
  workers = []
  for idx in range(config.n_worker):
    workers.append(mp.Process(target=worker_func, args=(idx,)))

  # Execute and wait for the end of the training
  for worker in workers:
    worker.start()

  for worker in workers:
    worker.join()

if __name__ == '__main__':
  tf.app.run()
