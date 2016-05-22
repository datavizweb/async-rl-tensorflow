import tensorflow as tf

from .ops import conv2d, linear, batch_sample

class AsyncNetwork(object):
  def __init__(self, data_format, history_length,
               screen_height, screen_width,
               action_size, activation_fn=tf.nn.relu,
               initializer=tf.truncated_normal_initializer(0, 0.02), name=None):
    self.w = {}
    self.t_w = {}

    with tf.variable_scope(name or 'nature'):
      with tf.variable_scope('dqn'):
        if data_format == 'NHWC':
          self.s_t = tf.placeholder('float32',
              [None, screen_width, screen_height, history_length], name='s_t')
        else:
          self.s_t = tf.placeholder('float32',
              [None, history_length, screen_width, screen_height], name='s_t')

        self.l0 = tf.div(self.s_t, 255.)

        self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.l0,
            32, [8, 8], [4, 4], initializer, activation_fn, data_format, name='l1_conv')
        self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
            64, [4, 4], [2, 2], initializer, activation_fn, data_format, name='l2_conv')
        self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
            64, [3, 3], [1, 1], initializer, activation_fn, data_format, name='l3_conv')

        self.l4, self.w['l4_w'], self.w['l4_b'] = \
            linear(self.l3, 512, activation_fn=activation_fn, name='l4_linear')

      # policy
      with tf.variable_scope('policy'):
        self.logits, self.w['p_w'], self.w['p_b'] = linear(self.l4, action_size, name='policy_linear')

        self.policy = tf.nn.softmax(self.logits, name='policy')
        self.action = tf.argmax(self.policy, dimension=1)

        self.log_policy = tf.nn.log_softmax(self.logits)

        self.sampled_actions = batch_sample(self.policy)
        self.log_policy_from_sampled_actions = tf.gather(self.log_policy, self.sampled_actions)

        self.entropy = -tf.reduce_sum(self.policy * self.log_policy, 1)

      # value function
      with tf.variable_scope('value'):
        self.value, self.w['q_w'], self.w['q_b'] = linear(self.l4, 1, name='value')

      with tf.variable_scope('optim'):
        self.target_reward = tf.placeholder([None], name='target_reward')

  def create_copy_ops(self, target):
    copy_ops = []

    with tf.variable_scope('copy'):
      for name in self.w.keys():
        copy_ops.append(target.w[name].assign(self.w[name]))

    return copy_ops
