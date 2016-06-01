import tensorflow as tf

class RMSPropOptimizer(tf.train.RMSPropOptimizer):
  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works
    var_list = [v for g, v in grads_and_vars if g is not None]
    with ops.control_dependencies(None):
      self._create_slots(var_list)
    update_ops = []
    with ops.op_scope([], name, self._name) as name:
      self._prepare()
      for grad, var in grads_and_vars:
        if grad is None:
          continue
        with ops.name_scope("update_" + var.op.name), ops.colocate_with(var):
          if isinstance(grad, ops.Tensor):
            update_ops.append(self._apply_dense(grad, var))
          else:
            update_ops.append(self._apply_sparse(grad, var))
      if global_step is None:
        return self._finish(update_ops, name)
      else:
        with ops.control_dependencies([self._finish(update_ops, "update")]):
          with ops.colocate_with(global_step):
            return state_ops.assign_add(global_step, 1, name=name).op

