import numpy as np
import os
import cv2

import numpy as np

def noisy(image):
  row, col = image.shape
  mean = 0
  var = 0.08
  sigma = var ** 0.5
  gauss = np.random.normal(mean, sigma, (row, col))
  return image + gauss