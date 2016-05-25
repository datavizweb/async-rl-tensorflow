import tensorflow as tf

def copy_deep_q_network(from_, to):
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
