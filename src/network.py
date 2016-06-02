import os
import tensorflow as tf

from .ops import conv2d, linear, batch_sample

class Network(object):
  def __init__(self, sess, data_format, history_length,
               screen_height, screen_width,
               action_size, activation_fn=tf.nn.relu,
               initializer=tf.truncated_normal_initializer(0, 0.02), 
               gamma=0.01, beta=0.0, global_network=None, global_optim=None):
    self.sess = sess

    if data_format == 'NHWC':
      self.s_t = tf.placeholder('float32',
          [None, screen_width, screen_height, history_length], name='s_t')
    elif data_format == 'NCHW':
      self.s_t = tf.placeholder('float32',
          [None, history_length, screen_width, screen_height], name='s_t')
    else:
      raise ValueError("unknown data_format : %s" % data_format)

    with tf.variable_scope('Nature_DQN'), tf.device('/cpu:0'):
      self.w = {}

      self.l0 = tf.div(self.s_t, 255.)
      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.l0,
          32, [8, 8], [4, 4], initializer, activation_fn, data_format, name='l1_conv')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          64, [4, 4], [2, 2], initializer, activation_fn, data_format, name='l2_conv')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
          64, [3, 3], [1, 1], initializer, activation_fn, data_format, name='l3_conv')

      self.l4, self.w['l4_w'], self.w['l4_b'] = \
          linear(self.l3, 512, activation_fn=activation_fn, name='l4_linear')

    with tf.variable_scope('policy'):
      # 512 -> action_size
      self.policy_logits, self.w['p_w'], self.w['p_b'] = linear(self.l4, action_size, name='linear')

      self.policy = tf.nn.softmax(self.policy_logits, name='pi')
      self.log_policy = tf.log(tf.nn.softmax(self.policy_logits))
      self.policy_entropy = -tf.reduce_sum(self.policy * self.log_policy, 1)
      _ = tf.scalar_summary('policy/entropy', self.policy_entropy)

      self.pred_action = tf.argmax(self.policy, dimension=1)

      self.sampled_action = batch_sample(self.policy)
      sampled_action_one_hot = tf.one_hot(self.sampled_action, action_size, 1., 0.)

      self.log_policy_of_sampled_action = tf.reduce_sum(self.log_policy * sampled_action_one_hot, 1)

    with tf.variable_scope('value'):
      # 512 -> 1
      self.value, self.w['q_w'], self.w['q_b'] = linear(self.l4, 1, name='linear')

    with tf.variable_scope('optim'):
      self.R = tf.placeholder('float32', [None], name='target_reward')

      self.true_log_policy = tf.placeholder('float32', [None], name='true_action')

      # TODO: equation on paper and codes of other implementations are different
      self.policy_loss = self.true_log_policy \
          * (self.R - self.value + beta * self.policy_entropy)
      self.value_loss = tf.pow(self.R - self.value, 2)

      self.total_loss = self.policy_loss + self.value_loss

    if global_network != None:
      with tf.variable_scope('copy_from_target'):
        copy_ops = []

        for name in self.w.keys():
          copy_op = self.w[name].assign(global_network.w[name])
          copy_ops.append(copy_op)

        self.global_copy_op = tf.group(*copy_ops, name='global_copy_op')

      # Add accumulated gradient ops for n-step Q-learning
      accum_grads, accum_grad_adds, reset_grads = [], [], []

  def copy_from_global(self):
    self.sess.run(self.global_copy_op)

  def save_model(self, saver, checkpoint_dir, step=None):
    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver.save(self.sess, checkpoint_dir, global_step=step)

  def load_model(self, saver, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      fname = os.path.join(checkpoint_dir, ckpt_name)
      saver.restore(self.sess, fname)
      print(" [*] Load SUCCESS: %s" % fname)
      return True
    else:
      print(" [!] Load FAILED: %s" % checkpoint_dir)
      return False
