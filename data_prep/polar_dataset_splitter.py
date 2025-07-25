import os.path

from scipy.io import loadmat
import numpy as np
import pickle
import datetime

def uv_to_mag_dir(tensor):
    u = tensor[:, :, 0, :]
    v = tensor[:, :, 1, :]

    magnitude = np.sqrt(u ** 2 + v ** 2)
    direction = np.arctan2(v, u)  # radians
    direction_rad = (direction + 2 * np.pi) % (2 * np.pi)

    # Combine into a new tensor: (lat, lon, 2, time)
    mag_dir = np.stack([magnitude, direction_rad], axis=2)
    return mag_dir

pycharm_dumb_flag = False
# pickle file is found in the project directory if you run it in pycharm, otherwise it might be in your desktop!

if(os.path.exists("../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat")):
    pycharm_dumb_flag = True
    file_name = "../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"
    mat_data = loadmat(file_name)
else :
    file_name = "data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"
    mat_data = loadmat(file_name)

print("splitting:", file_name)

u_tensors = mat_data['u']
v_tensors = mat_data['v']

u_tensors = np.expand_dims(u_tensors,axis=2)
v_tensors = np.expand_dims(v_tensors,axis=2)

all_tensors = np.append(u_tensors,v_tensors,2)


# Creating tensors for training, validating, testing subsets of data
training_data = np.zeros([len(all_tensors),len(all_tensors[0]),len(all_tensors[0][0]),1])
validation_data = np.zeros([len(all_tensors),len(all_tensors[0]),len(all_tensors[0][0]),1])
test_data = np.zeros([len(all_tensors),len(all_tensors[0]),len(all_tensors[0][0]),1])

for i in range(0, len(all_tensors[0][0][0]), 130):
    training_data = np.append(training_data, all_tensors[::, ::, ::, i:i+70], 3)
    validation_data = np.append(validation_data, all_tensors[::, ::, ::, i+80:i + 95], 3)
    test_data = np.append(test_data, all_tensors[::, ::, ::, i+105:i + 120], 3)

training_data = uv_to_mag_dir(training_data)
validation_data = uv_to_mag_dir(validation_data)
test_data = uv_to_mag_dir(test_data)

meta_data = {
    'date_made': datetime.datetime.now(),
    'data_split_strat': (
        'Alternating chunks of 120 frames go into training, validation, and test. '
        'The first 70 go into training, space of 10, 15 in validation, space of 10, '
        'last 15 go into test, space of 10'
    ),
    'dimension_data': 'lat, lon, <magnitude><direction_radians>, time'
}

this_is_a_dictionary = {
                        'training_data':training_data,
                        'validation_data':validation_data,
                        'test_data':test_data,
                        'meta_data':meta_data
                        }

# Now the training data holds [magnitude, direction_rad], not [u, v]
magnitude = this_is_a_dictionary['training_data'][:, :, 0, :]
direction = this_is_a_dictionary['training_data'][:, :, 1, :]

avg_magnitude = np.nanmean(magnitude)
avg_direction = np.nanmean(direction)  # in radians

print("Average vector magnitude in training data:", avg_magnitude)
print("Average direction (radians) in training data:", avg_direction)

print("New training data shape:", training_data.shape)
print("Sample magnitude range:", np.nanmin(training_data[:,:,0,:]), "to", np.nanmax(training_data[:,:,0,:]))
print("Sample direction range:", np.nanmin(training_data[:,:,1,:]), "to", np.nanmax(training_data[:,:,1,:]))


try:
    path = '../polarized_data.pickle' if pycharm_dumb_flag else 'polarized_data.pickle'
    with open(path, 'wb') as file:
        pickle.dump([training_data, validation_data, test_data], file)
    print(f"Pickle saved to: {path}")
except Exception as e:
    print("Failed to pickle:", e)

print ("you've been super pickle'd")