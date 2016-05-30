import os
import tensorflow as tf

try:
  import cv2
  imresize = cv2.resize
  imwrite = cv2.imwrite
except:
  import scipy.misc
  imresize = scipy.misc.imresize
  imwrite = scipy.misc.imsave

def save_history_as_image(prefix, history):
  for idx in range(0, 4):
    imwrite("%s_%d.png" % (prefix, idx), history[:,:,idx])

def accumulate_gradients(tower_grads):
  accumulate_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, var in grad_and_vars:
      if g is not None:
        expanded_g = tf.expand_dims(g, 0)
        grads.append(expanded_g)
      else:
        continue

    if grads:
      grad = tf.concat(0, grads)
      grad = tf.reduce_sum(grad, 0)

      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      accumulate_grads.append(grad_and_var)
  return accumulate_grads

def make_checkpoint_dir(config):
  keys = [
    'data_format', 'env_name', 'n_action_repeat', 'screen_height',
    'screen_width', 'history_length',
  ]
  names = []
  for k, v in config.__dict__['__flags'].items():
    if k in keys:
      names.append("%s=%s" % (k, v))
  return os.path.join('checkpoints', *names) + '/'
