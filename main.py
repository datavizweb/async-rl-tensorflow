import gym
import random
import logging
import tensorflow as tf
from threading import Thread

from models.a3c import A3C_FF
from models.deep_q_network import DeepQNetwork

flags = tf.app.flags

# Deep Q Network
flags.DEFINE_string('data_format', 'NCHW', 'The format of convolutional filter')

# Environment
flags.DEFINE_string('env_name', 'SpaceInvaders-v0', 'The name of gym environment to use')
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
flags.DEFINE_float('gamma', 0.0, 'Discount factor of return')
flags.DEFINE_float('beta', 0.0, 'Beta of RMSProp optimizer')
flags.DEFINE_integer('t_max', 100000, 'The maximum number of t while training')
flags.DEFINE_integer('n_step', 5, 'The maximum number of n')
flags.DEFINE_integer('n_thread', 2, 'The number of threads to run asynchronously')

# Debug
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_string('log_level', 'INFO', 'Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

config = flags.FLAGS

logger = logging.getLogger()
logger.propagate = False

logger.setLevel(config.log_level)

# Set random seed
tf.set_random_seed(config.random_seed)
random.seed(config.random_seed)

def main(_):
  with tf.Session() as sess:
    global_network = DeepQNetwork(config.data_format,
                                  config.history_length,
                                  config.screen_height,
                                  config.screen_width,
                                  gym.make(config.env_name).action_space.n)
    global_optim = tf.train.RMSPropOptimizer(config.learning_rate, config.decay, config.momentum, config.epsilon)

    global_t = 0
    thread_stop = False
    def train_function(idx):
      model = models[idx]
      state, reward, terminal = model.env.new_random_game()

      while True:
        diff_global_t = model.act(state, reward, terminal)
        global_t += diff_global_t

    models = []
    for thread_id in range(config.n_thread):
      model = A3C_FF(thread_id, config, sess, global_network, global_optim)
      models.append(model)

    tf.initialize_all_variables().run()
    saver = tf.train.Saver(global_network.w.values(), max_to_keep=30)

    for model in models:
      model.copy_from_global()

    threads = []
    for idx in range(config.n_thread):
      threads.append(Thread(target=train_function, args=(idx,)))

    for thread in threads:
      thread.start()

    for thread in threads:
      thread.join()

if __name__ == '__main__':
  tf.app.run()
