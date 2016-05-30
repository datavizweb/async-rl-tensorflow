import gym
import time
import random
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from threading import Thread
import multiprocessing as mp

from models.a3c import A3C_FF
from models.utils import make_checkpoint_dir
from models.deep_q_network import DeepQNetwork

flags = tf.app.flags

# Deep Q Network
flags.DEFINE_string('data_format', 'NHWC', 'The format of convolutional filter')
flags.DEFINE_string('ep_start', 1., 'The value of epsilon at start in e-greedy')
flags.DEFINE_string('ep_end', 0.1, 'The value of epsilnon at the end in e-greedy')
flags.DEFINE_string('ep_end_t', 10000000, 'The time t when epsilon reach ep_end')

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
flags.DEFINE_string('t_test', 100, 'The time t when epsilon reach ep_end')
flags.DEFINE_string('t_save', 100000, 'The time t when epsilon reach ep_end')
flags.DEFINE_string('t_end', 10000000, 'The time t when epsilon reach ep_end')
flags.DEFINE_integer('n_worker', 4, 'The number of threads to run asynchronously')
flags.DEFINE_boolean('use_thread', True, 'Whether to use thread or process for each worker')

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

if config.use_thread:
  t_global = 0
  should_stop = False
else:
  counter = mp.Value('l', 0)
  should_stop = mp.Value('l', False)

def main(_):
  if config.use_thread:
    global t_global

  #with tf.Session() as sess:
  with tf.Session(config=tf.ConfigProto(
      intra_op_parallelism_threads=1)) as sess:
    with tf.variable_scope('master') as scope:
      global_network = DeepQNetwork(sess, config.data_format,
                                    config.history_length,
                                    config.screen_height,
                                    config.screen_width,
                                    gym.make(config.env_name).action_space.n)
      global_optim = tf.train.RMSPropOptimizer(config.learning_rate, config.decay, config.momentum, config.epsilon)

    # Define thread-specific models
    models = []
    for worker_id in range(config.n_worker):
      model = A3C_FF(worker_id, config, sess, global_network, global_optim)
      models.append(model)

    t_global_op = tf.Variable(0, trainable=False, name='t_global')
    t_global_input = tf.placeholder('int32', name='t_global_input')
    t_global_assign = t_global_op.assign(t_global_input)

    logger.info("Initialize and load model weights")

    tf.initialize_all_variables().run()

    saver = tf.train.Saver(global_network.w.values() + [t_global_op], max_to_keep=30)
    checkpoint_dir = make_checkpoint_dir(config)

    global_network.load_model(saver, checkpoint_dir)

    if config.use_thread:
      t_global = t_global_op.eval()
    else:
      counter.value = t_global_op.eval()

    logger.info("Copy weights from the global models")

    for model in models:
      model.copy_from_global()

    def train_function(worker_id):
      if config.use_thread:
        global t_global, should_stop
      else:
        t_global = counter.value

      model = models[worker_id]
      state, reward, terminal = model.env.new_random_game()

      idx = 0
      start_time = time.time()

      while True:
        idx += 1

        if t_global > config.t_end:
          break

        # 1. predict
        action = model.predict(state)
        # 2. act
        state, reward, terminal = model.env.step(action, is_training=True)
        # 3. observe
        model.observe(state, reward, terminal)

        if terminal:
          state, reward, terminal = model.env.new_random_game()

        if config.use_thread:
          t_global += 1
        else:
          with counter.get_lock():
            counter.value += 1
            t_global = counter.value

        # Test
        if t_global % config.t_test == config.t_test - 1:
          current_time = time.time()
          print worker_id, idx / (current_time - start_time)
          idx, start_time = 0, current_time

        # Job only for the first worker
        if worker_id == 0:
          if t_global % config.t_save == config.t_save - 1:
            # save
            sess.run(t_global_assign, {t_global_input: t_global})
            global_network.save_model(saver, checkpoint_dir, step=t_global)

    # Test for signle thread
    # train_function(0)

    # Prepare each threads to run asynchronously
    workers = []
    for worker_id in range(config.n_worker):
      if config.use_thread:
        workers.append(Thread(target=train_function, args=(worker_id,)))
      else:
        workers.append(mp.Process(target=train_function, args=(worker_id,)))

    logger.info("Start workers")

    # Execute and wait for the end of the training
    for worker in workers:
      worker.start()

    for worker in workers:
      worker.join()

if __name__ == '__main__':
  tf.app.run()
