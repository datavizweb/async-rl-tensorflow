def get_config(FLAGS):
  if FLAGS.model == 'm1':
    pass

  for k, v in FLAGS.__dict__['__flags'].items():
    if hasattr(config, k):
      setattr(config, k, v)

  return config
