#Ember McEwen, last edit 4:37 pm on 2024.6.4

import scipy.io as sio
from PIL import Image
import numpy as np
import datetime

#17040 94x44 images plus some other data
mat_contents = sio.loadmat('./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2')

u_array = mat_contents['u']
v_array = mat_contents['v']
time = mat_contents['ocean_time'].squeeze()
time_array = [datetime.datetime.fromordinal(int(t)) + datetime.timedelta(days=t % 1) - datetime.timedelta(days=366) for t
              in time]

#previously found using the normal method
minU, maxU = -0.8973235906436031, 1.0859991093945718
minV, maxV = -0.6647028130174489, 0.5259408400292674

#normalize between 0 and 255
adjusted_u_arr = 255 * (u_array - minU) / (maxU - minU)
adjusted_v_arr = 255 * (v_array - minV) / (maxV - minV)

img = Image.new('RGB', (94, 44), color='white')


def generate_image(image_num):

    #time = str(time_array[image_num]))

    for y in range(94):
        for x in range(44):
            is_land = False
            if np.isnan(adjusted_u_arr[y][x][image_num]):
                #when there was no u value the minimal dataloader displayed it is land
                curr_u = 0
                is_land = True
            else:
                curr_u = int(adjusted_u_arr[y][x][image_num])
            if np.isnan(adjusted_v_arr[y][x][0]):
                #when there was no v value it was shown as a place in the ocean with no currents
                curr_v = 0
                curr_u = 0
            else:
                curr_v = int(adjusted_v_arr[y][x][image_num])

            img.putpixel((y, 43 - x), (curr_u, curr_v, is_land * 255))

    return img
    #replace return with save or show if desired

generate_image(10).show()
