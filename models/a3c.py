import tensorflow as tf

from .deep_q_network import AsyncNetwork

def update_target_q_network(from_, to):
  for name in w.keys():
    t_w_assign_op[name].eval({t_w_input[name]: w[name].eval()})

def accumulate_gradients(tower_grads):
  accumulate_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    grad = tf.concat(0, grads)
    grad = tf.reduce_sum(grad, 0)

    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    accumulate_grads.append(grad_and_var)
  return accumulate_grads

class A3C_FF(object):
  def __init__(self, global_model, optim):
    self.global_model = global_model
    self.optim = optim

    self.model = model(config)

    self.t = 0
    self.t_start = 0
    self.t_max = t_max
    self.learning_rate = learning_rate

    self.s_ts = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype = np.float16)
    self.rewards = np.empty(self.memory_size, dtype = np.integer)
    self.values = np.empty(self.memory_size, dtype = np.integer)

    copy_weights(self.global_model, self.model)

  def build_model(self):
    self.networks, grads = [], []

    for step in xrange(self.max_step):
      with tf.name_scpe('A3C_%d' % step) as scope:
        network = AsyncNetwork()
        self.networks.append(network)

        # Share parameters between networks
        tf.get_variable_scope().reuse_variables()

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

        grad = self.optim.compute_gradients(network.total_loss)
        grads.append(grad)

    # Accumulate gradients for n-steps
    accumulated_grads = accumulate_gradients(grads)

    for grad, var in accumulated_grads:
      if grad is not None:
        summaries.append(
            tf.histogram_summary('p%d/%s/gradients' % var.op.name, grad))

    self.optim_step = tf.placeholder('int32', None, name='optim_step')

    self.apply_gradeint_op = self.optim.apply_gradients(
        self.shared_model.variables, accumulated_grads, global_step=self.optim_step)

    if self.pid == 0:
      for var in tf.trainable_variables():
        summaries.append(tf.histogram_summary('p%d/%d' % (self.pid, var.op.name), var))

      # Track the moving averages of all trainable variables.
      variable_averages = tf.train.ExponentialMovingAverage(
          cifar10.MOVING_AVERAGE_DECAY, global_step)
      variables_averages_op = variable_averages.apply(tf.trainable_variables())

      self.train_op = tf.group(
          self.policy_apply_gradeint_op, self.value_apply_gradient_op, variables_averages_op)

      # Create a saver.
      saver = tf.train.Saver(tf.all_variables())

  def act(self, s_t, reward, terminal):
    self.learning_rate = (max_step - global_t - 1) / max_step * args.learning_rate

    # clip reward
    if self.max_reward:
      reward = min(self.max_reward, reward)
    if self.min_reward:
      reward = max(self.min_reward, reward)

    self.prev_rewards[self.step - 1] = reward

    if (terminal and self.t_start < self.t) or self.t - self.t_start == self.t_max:
      r, a, s = {}, {}, {t: s_t}
      if terminal:
        r[t] = 0
      else:
        _, r[t] = self.model.calc_policy_value(s_t)

      for t in xrange(self.t_start, self.t, -1):
        r[t] = self.prev_r[t] + self.gamma * r[t]

      data1 = {network.s_t: self.prev_s[t] for network in self.networks}
      data2 = {network.R: r[t] for network in self.networks}

      self.sess.run(self.self.train_op, feed_dict = {**data1, **data2})
    else:
      Q = self.sess.run([self.model.value], {self.model.s_t: s_t})

    if not terminal:
      self.prev_s[self.t] = s_t
      self.prev_r[self.t] = reward
      self.prev_t[self.t] = terminal
      self.t += 1
    else:
      print "Should implement"
      pass
