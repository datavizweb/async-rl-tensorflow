import random
import logging
import tensorflow as tf

from models.environment import Environment
from models.deep_q_network import NatureDQN

flags = tf.app.flags

# Hardware
flags.DEFINE_boolean('cpu', False, 'Whether to use cpu for convolution')

# Environment
flags.DEFINE_string('env_name', 'SpaceInvaders-v0', 'The name of gym environment to use')
flags.DEFINE_integer('n_action_repeat', 4, 'The number of actions to repeat')
flags.DEFINE_integer('max_random_start', 30, 'The maximum number of NOOP actions at the beginning of an episode')
flags.DEFINE_integer('screen_height', 84, 'The height of gym screen')
flags.DEFINE_integer('screen_width', 84, 'The width of gym screen')
flags.DEFINE_integer('history_length', 4, 'The length of history of screens to use as an input to DQN')

# Training
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_float('learning_rate', 0.0032, 'The learning rate of training')

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
    if config.cpu:
      data_format = 'NHWC'
    else:
      data_format = 'NCHW'

    env = Environment(config.env_name, config.n_action_repeat, config.max_random_start,
        config.history_length, config.screen_height, config.screen_width)

    dqn = NatureDQN(data_format, config.history_length,
        config.screen_height, config.screen_width, env.action_size, name='nature')

    agent = Agent(config, env, sess)

    if config.is_train:
      agent.train()
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
