import tensorflow as tf

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
