import random
import logging
import tensorflow as tf

from models.environment import Environment
from models.a3c import A3C_FF

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
flags.DEFINE_integer('n_process', 16, 'The number of processes to run asynchronously')

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

    global_model = AsyncNetwork()
    global_optim = tf.train.RMSPropOptimzer()

    agent = A3C_FF(config, global_model, global_optim, sess)

    threads = []
    global_t = 0

    def train_function(idx):
      global global_t
      
      thread = threads[idx]

      while True:
        if thread_stop:
          break
        if global_t > config.global_t_max:
          break

        diff_global_t = thread.process(sess, global_t, summary_writer, summary_op, score_input)
        global_t += diff_global_t

    threads = []
    for idx in range(self.n_process):
      thread = A3C_FF(idx, global_network, initial_learning_rate,
                      learning_rate_input,
                      grad_applier, MAX_TIME_STEP,
                      device = device)
      threads.append(training_thread)

    for idx in range(self.n_process):
      threads.append(threading.Thread(target=train_function, args=(idx,)))

if __name__ == '__main__':
  tf.app.run()
