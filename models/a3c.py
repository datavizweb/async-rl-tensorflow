import copy

class A3C(object):
  def __init__(self):
    pass

  def build_model(self):
    def make_model():
      model = A3C_FF()
      optim = tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon).minimize(model.cost)

      return model, optim

    def func(process_id):
      env = Environment(config.env_name, config.n_action_repeat, config.max_random_start,
          config.history_length, config.screen_height, config.screen_width)

      model, optim = make_model()

      dqn = NatureDQN(data_format, config.history_length,
          config.screen_height, config.screen_width, env.action_size, name='nature')
