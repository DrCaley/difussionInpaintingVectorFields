import os.path

from scipy.io import loadmat
import numpy as np
import pickle
import datetime

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

training_data = training_data[::, ::, ::, 1:]
validation_data = validation_data[::, ::, ::, 1:]
test_data = test_data[::, ::, ::, 1:]

meta_data = {'date_made':datetime.datetime.now(),
            'data_split_strat':'Alternating chucks of 120 frames go into training, validation'
                             'and test. The first 70 go into training, space of 10, 15 in'
                             ' validation, space of 10, last 15 go into test, space of 10',
            'dimension_data':'lat,long,<u><v>,time'}

this_is_a_dictionary = {'training_data':training_data,
                        'validation_data':validation_data,
                        'test_data':test_data,
                        'meta_data':meta_data}

u_values = this_is_a_dictionary['training_data'][:, :, 0, :].reshape(-1)
v_values = this_is_a_dictionary['training_data'][:, :, 1, :].reshape(-1)

u_training_mean = np.nanmean(u_values)
u_training_std = np.nanstd(u_values)
v_training_mean = np.nanmean(v_values)
v_training_std = np.nanstd(v_values)

try:
    path = '../data.pickle' if pycharm_dumb_flag else 'data.pickle'
    with open(path, 'wb') as file:
        pickle.dump([training_data, validation_data, test_data], file)
    print(f"Pickle saved to: {path}")
except Exception as e:
    print("Failed to pickle:", e)

print ("you've been pickle'd")