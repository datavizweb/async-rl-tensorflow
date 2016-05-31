import os
import time
import tensorflow as tf
from logging import getLogger

logger = getLogger(__name__)

try:
  import cv2
  imresize = cv2.resize
  imwrite = cv2.imwrite
except:
  import scipy.misc
  imresize = scipy.misc.imresize
  imwrite = scipy.misc.imsave

def timeit(f):
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    logger.info("%s : %2.2f sec" % (f.__name__, end_time - start_time))
    return result
  return timed
