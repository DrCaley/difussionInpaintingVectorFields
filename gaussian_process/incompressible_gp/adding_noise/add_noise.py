import numpy as np
import os
import cv2
import torch

from PIL import Image
from gauss_random_sines import noisy
from divergence_free_noise import curl_of_gradient_noise

"""
1 is white, 0 is black
"""

blank_tensor0 = torch.zeros(1024,1024)
blank_array0 = (blank_tensor0.numpy()).astype(np.uint8)

noisy_image = noisy(blank_array0)

divergence_free = curl_of_gradient_noise((1024, 1024))
noisy_image = noisy_image * divergence_free

noisy_image = noisy_image * 255
noisy_image_display0 = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)



blank_tensor1 = torch.ones(1024,1024)
blank_array1 = (blank_tensor1.numpy()).astype(np.uint8)
noisy_image_display1 = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)



cv2.imshow("Noisy Image Zero", noisy_image_display0 - noisy_image_display1)
cv2.waitKey(0)
cv2.destroyAllWindows()