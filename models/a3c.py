from .base import BaseModel

class A3C(BaseModel):
  def __inti__(self, config, sess):
    super(A3C, self).__init__(self, config)

  def build_model(self):
    pass
