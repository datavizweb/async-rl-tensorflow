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
