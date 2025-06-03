import numpy as np
import os
import cv2
import torch

from PIL import Image
from gauss_random_sines import noisy

"""
1 is white, 0 is black
"""
blank_tensor = torch.ones(1024,1024)
blank_array = (blank_tensor.numpy()).astype(np.uint8)

noisy_image = noisy(blank_array)
noisy_image = noisy_image * 255
noisy_image_display = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)

cv2.imshow("Noisy Image", noisy_image_display)
cv2.waitKey(0)
cv2.destroyAllWindows()