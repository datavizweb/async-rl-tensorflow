import tensorflow as tf

def update_target_q_network(from_, to):
  for name in w.keys():
    t_w_assign_op[name].eval({t_w_input[name]: w[name].eval()})

class A3C(object):
  def __init__(self, config):
    self.global_model = model(config)
    self.model = model(config)

    self.states = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype = np.float16)
    self.rewards = np.empty(self.memory_size, dtype = np.integer)
    self.values = np.empty(self.memory_size, dtype = np.integer)

    copy_weights(self.global_model, self.model)

  def act(self, state, reward, terminal):
    # clip reward
    if self.max_reward:
      reward = min(self.max_reward, reward)
    if self.min_reward:
      reward = max(self.min_reward, reward)

    self.prev_rewards[self.step - 1] = reward

    if terminal:
      R = 0
    else:
      Q = self.sess.run([self.model.value], {self.model.s_t: state})

    policy_loss = 0
    value_loss = 0

    for t in xrange(self.t_start, self.t, -1):
      R = self.prev_r[t] + self.gamma * R
      Q = self.prev_Q[t]

      self.log_policy[t] * (

      diff = R - Q
