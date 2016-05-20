import tensorflow as tf

from .ops import conv2d, linear

class NatureDQN(object):
  def __init__(self, history_length, data_format
               screen_height, screen_width
               action_size, activation_fn=tf.nn.relu,
               initializer=tf.truncated_normal_initializer(0, 0.02), name=None):
    self.w = {}
    self.t_w = {}

    with tf.variable_scope(name or 'nature'):
      if data_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, screen_width, screen_height, history_length], name='s_t')
      else:
        self.s_t = tf.placeholder('float32',
            [None, history_length, screen_width, screen_height], name='s_t')

      self.l0 = tf.div(self.s_t, 255.)

      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.l0,
          32, [8, 8], [4, 4], initializer, activation_fn, data_format, name='l1')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          64, [4, 4], [2, 2], initializer, activation_fn, data_format, name='l2')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
          64, [3, 3], [1, 1], initializer, activation_fn, data_format, name='l3')

      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
      self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.action_size, name='q')
      self.q_action = tf.argmax(self.q, dimension=1)

  def create_copy_ops(self, target):
    self.copy_ops = []
    with tf.variable_scope('copy'):
      for name in self.w.keys():
        self.copy_ops.append(target.w[name].assign(self.w[name]))
