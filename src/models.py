import re
import time
import random
import numpy as np
import tensorflow as tf

from .utils import range, logger

expand = lambda s_t: np.expand_dims(s_t, 0)

class A3C_FF(object):
  def __init__(self, worker_id, sess, local_networks, local_env,
               global_network, global_optim, config, learning_rate_op):
    self.sess = sess
    self.worker_id = worker_id
    self.writer = None

    self.networks = local_networks
    self.env = local_env

    self.global_network = global_network
    self.global_optim = global_optim

    self.t = 0
    self.t_start = 0
    self.t_max = config.t_max
    self.t_save = config.t_save
    self.t_test = config.t_test
    self.t_train_max = config.t_train_max

    self.ep_start = config.ep_start
    self.ep_end = config.ep_end
    self.ep_end_t = config.ep_end_t

    self.learning_rate = config.learning_rate
    self.gamma = config.gamma
    self.max_reward = config.max_reward
    self.min_reward = config.min_reward
    self.max_grad_norm = config.max_grad_norm

    self.data_format = config.data_format
    self.screen_height = config.screen_height
    self.screen_width = config.screen_width
    self.history_length = config.history_length

    self.prev_r = {}

    self.s_t_shape = self.networks[0].s_t.get_shape().as_list()
    self.s_t_shape[0] = 1

    self.make_accumulated_gradients()

    if self.worker_id == 0:
      with tf.variable_scope('summary'):
        scalar_summary_tags = ['average/reward', \
            'episode/max reward', 'episode/min reward', 'episode/avg reward', 'episode/num of game']

        self.summary_placeholders = {}
        self.summary_ops = {}

        for tag in scalar_summary_tags:
          self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
          self.summary_ops[tag] = tf.scalar_summary(tag, self.summary_placeholders[tag])

        histogram_summary_tags = ['episode/rewards', 'episode/actions']

        for tag in histogram_summary_tags:
          self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
          self.summary_ops[tag] = tf.histogram_summary(tag, self.summary_placeholders[tag])

        summaries = []
        avg_value = tf.reduce_mean(self.networks[0].value)
        summaries.append(tf.scalar_summary('value/', avg_value))

        avg_policy = tf.reduce_mean(self.networks[0].policy, 0)
        for idx in range(self.env.action_size):
          summaries.append(tf.scalar_summary('policy/%s' % idx, avg_policy[idx]))
        summaries.append(tf.scalar_summary('policy/entropy',
            tf.reduce_mean(self.networks[0].policy_entropy)))
        summaries.append(tf.scalar_summary('loss/value_loss',
            tf.reduce_mean(self.networks[0].value_loss)))
        summaries.append(tf.scalar_summary('loss/policy_loss',
            tf.reduce_mean(self.networks[0].policy_loss)))
        summaries.append(tf.scalar_summary('loss/total_loss',
            tf.reduce_mean(self.networks[0].total_loss)))
        summaries.append(tf.scalar_summary('train/learning_rate',
            learning_rate_op))

        self.value_policy_summary = tf.merge_summary(summaries, 'policy_summary')

  def train(self, global_t):
    # 0. Prepare training
    state, reward, terminal = self.env.new_random_game()
    self.observe(state, reward, terminal)

    start_time = time.time()
    while True:
      if global_t[0] > self.t_train_max:
        break

      # 1. Predict
      action = self.predict(state)
      # 2. Step
      state, reward, terminal = self.env.step(-1, is_training=True)
      # 3. Observe
      self.observe(state, reward, terminal)

      if terminal:
        self.env.new_random_game()

      global_t[0] += 1

    logger.info("loop : %2.2f sec" % (time.time() - start_time))

  def train_with_log(self, global_t, saver, writer, checkpoint_dir,
                     assign_learning_rate_op, assign_global_t_op):
    from tqdm import tqdm

    self.writer = writer
    self.global_t = global_t

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward = 0
    max_avg_ep_reward = 0
    ep_rewards, actions = [], []

    # 0. Prepare training
    state, reward, terminal = self.env.new_random_game()
    self.observe(state, reward, terminal)

    start_time = time.time()

    for _ in tqdm(range(int(self.t_train_max)), ncols=70, initial=int(global_t[0])):
      if global_t[0] > self.t_train_max:
        break

      # 1. Predict
      action = self.predict(state)
      # 2. Step
      state, reward, terminal = self.env.step(-1, is_training=True)
      # 3. Observe
      self.observe(state, reward, terminal)

      global_t[0] += 1
      assign_learning_rate_op((self.t_train_max - global_t[0] + 1) / self.t_train_max * self.learning_rate)

      if terminal:
        self.env.new_random_game()

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        ep_reward += reward

      actions.append(action)
      total_reward += reward

      if self.t % self.t_test == self.t_test - 1:
        avg_reward = total_reward / self.t_test

        try:
          max_ep_reward = np.max(ep_rewards)
          min_ep_reward = np.min(ep_rewards)
          avg_ep_reward = np.mean(ep_rewards)
        except:
          max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

        logger.info('\nglobal_t: %d, avg_r: %.4f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
          % (global_t[0], avg_reward, avg_ep_reward, max_ep_reward, min_ep_reward, num_game))

        self.inject_summary({
            'average/reward': avg_reward,
            'episode/max reward': max_ep_reward,
            'episode/min reward': min_ep_reward,
            'episode/avg reward': avg_ep_reward,
            'episode/num of game': num_game,
            'episode/rewards': ep_rewards,
            'episode/actions': actions,
          })

        if self.t % self.t_save == self.t_save - 1:
          assign_global_t_op(global_t[0])
          self.global_network.save_model(saver, checkpoint_dir, step=global_t[0])

          max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

        num_game = 0
        total_reward = 0.
        ep_reward = 0.
        ep_rewards = []
        actions = []

    print("loop : %2.2f sec" % (time.time() - start_time))

  # Add accumulated gradient ops for n-step Q-learning
  def make_accumulated_gradients(self):
    global_var = {w.name.replace('A3C_global/', ''):w for w in self.global_network.w.values()}

    reset_accum_grads = []
    new_grads_and_vars = []

    # 1. Prepare accum_grads
    self.accum_grads = {}
    self.add_accum_grads = {}

    for step, network in enumerate(self.networks):
      grads_and_vars = self.global_optim.compute_gradients(network.total_loss, network.w.values())
      _add_accum_grads = []

      for grad, var in tuple(grads_and_vars):
        if grad is not None:
          shape = grad.get_shape().as_list()

          name = 'accum/%s' % "/".join(var.name.split(':')[0].split('/')[-3:])
          if step == 0:
            self.accum_grads[name] = tf.Variable(
                tf.zeros(shape), trainable=False, name=name)

            global_v = global_var[re.sub(r'.*\/A3C_\d+\/', '', var.name)]
            new_grads_and_vars.append((tf.clip_by_norm(self.accum_grads[name].ref(), self.max_grad_norm), global_v))

            reset_accum_grads.append(self.accum_grads[name].assign(tf.zeros(shape)))

          _add_accum_grads.append(tf.assign_add(self.accum_grads[name], grad))

      # 2. Add gradient to accum_grads
      self.add_accum_grads[step] = tf.group(*_add_accum_grads)

    # 3. Reset accum_grads
    self.reset_accum_grad = tf.group(*reset_accum_grads)

    # 4. Update variables of global_network with accum_grads
    self.apply_gradient = self.global_optim.apply_gradients(new_grads_and_vars)

    for step, add_accum_grads in self.add_accum_grads.items():
      with tf.control_dependencies([add_accum_grads]):
        self.add_accum_grads[step] = tf.constant(0)

    # Add dummy_op to execute optimizer with partial_run
    with tf.control_dependencies([self.apply_gradient]):
      self.fake_apply_gradient = tf.constant(0)

  def reset_partial_graph(self):
    targets = [network.sampled_action for network in self.networks]
    targets.extend([network.value for network in self.networks])
    targets.extend(self.add_accum_grads.values())
    targets.append(self.fake_apply_gradient)

    if self.writer:
      targets.append(self.value_policy_summary)

    inputs = [network.s_t for network in self.networks]
    inputs.extend([network.R for network in self.networks])

    self.partial_graph = self.sess.partial_run_setup(targets, inputs)

  def predict(self, s_t, test_ep=None):
    if self.t_start - self.t == 0:
      self.reset_partial_graph()

    network_idx = self.t - self.t_start

    action = self.sess.partial_run(
      self.partial_graph,
      [
        self.networks[network_idx].sampled_action,
      ],
      {
        self.networks[network_idx].s_t: [s_t]
      }
    )
    action = action[0]
    self.t += 1

    return action

  def observe(self, s_t, r_t, terminal):
    self.prev_r[self.t] = max(self.min_reward, min(self.max_reward, r_t))

    if (terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
      r = {}

      if terminal:
        r[self.t] = 0.
      else:
        r[self.t] = self.sess.partial_run(
            self.partial_graph,
            self.networks[self.t_start - self.t].value,
        )[0][0]

      for t in range(self.t - 1, self.t_start - 1, -1):
        r[t] = self.prev_r[t] + self.gamma * r[t + 1]

      data = {}
      data.update({
        self.networks[t].R: [r[t + self.t_start]] for t in range(len(self.prev_r) - 1)
      })

      # 1. Update accumulated gradients
      if not self.writer:
        self.sess.partial_run(self.partial_graph,
            [self.add_accum_grads[t] for t in range(len(self.prev_r) - 1)], data)
      else:
        results = self.sess.partial_run(self.partial_graph,
            [self.value_policy_summary] + [self.add_accum_grads[t] for t in range(len(self.prev_r) - 1)], data)

        summary_str = results[0]
        self.writer.add_summary(summary_str, self.global_t[0])

      # 2. Update global w with accumulated gradients
      self.sess.run(self.apply_gradient)

      # 3. Reset accumulated gradients to zero
      self.sess.run(self.reset_accum_grad)

      # 4. Copy weights of global_network to local_network
      self.networks[0].copy_from_global()

      self.prev_r = {self.t: self.prev_r[self.t]}
      self.t_start = self.t

  def inject_summary(self, tag_dict):
    summary_str_lists = self.sess.run(
        self.networks[0].filter_summaries + [self.summary_ops[tag] for tag in tag_dict.keys()
      ], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.global_t[0])
