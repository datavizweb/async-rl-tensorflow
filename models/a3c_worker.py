import tensorflow as tf

def update_target_q_network(from_, to):
  for name in w.keys():
    t_w_assign_op[name].eval({t_w_input[name]: w[name].eval()})

class A3C_worker(object):
  def __init__(self):
    self.global_model = model(config)
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
    policy_grads, value_grads = [], []
    for step in xrange(self.max_step):
      with tf.name_scpe('A3C_%d' % step) as scope:

        tf.get_variable_scope().reuse_variables()

        policy_grad = self.policy_optim.compute_gradients(self.policy_loss)
        value_grad = self.value_optim.compute_gradients(self.value_loss)

        policy_grads.append(policy_grad)
        value_grads.append(value_grad)

        accumulated_grads = []
        for grads in [policy_grads, value_grad]:
          for grad_and_vars in zip(*grads):

  def act(self, s_t, reward, terminal):
    self.learning_rate = (max_step - global_t - 1) / max_step * args.learning_rate

    # clip reward
    if self.max_reward:
      reward = min(self.max_reward, reward)
    if self.min_reward:
      reward = max(self.min_reward, reward)

    self.prev_rewards[self.step - 1] = reward

    if terminal:
      R = 0

      policy_loss = 0
      value_loss = 0

      for t in xrange(self.t_start, self.t, -1):
        R = self.prev_r[t] + self.gamma * R
        Q = self.prev_Q[t]

        diff = R - Q
    else:
      Q = self.sess.run([self.model.value], {self.model.s_t: s_t})

    if terminal:
      self.model.reset_state()
      action = None
