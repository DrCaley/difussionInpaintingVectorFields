import os.path

from scipy.io import loadmat
import numpy as np
import pickle
import datetime
import h5py

# ── Set the input file — format is auto-detected from the extension ───────────
INPUT_FILE = "data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"

pycharm_dumb_flag = False

def resolve_path(path):
    """Try ../path first (PyCharm), then path as-is."""
    alt = os.path.join("..", path)
    if os.path.exists(alt):
        return alt, True
    return path, False

def load_dataset(input_file):
    file_path, is_pycharm = resolve_path(input_file)
    ext = os.path.splitext(file_path)[1].lower()
    print(f"splitting: {file_path}  (detected format: {ext})")

    if ext in ('.pkl', '.pickle'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # Pickle stores (T, H, W); transpose to (W, H, T) to match mat convention
        u = np.transpose(data['us'], (2, 1, 0))
        v = np.transpose(data['vs'], (2, 1, 0))
        return u, v, is_pycharm

    elif ext == '.mat':
        mat_data = loadmat(file_path)
        # mat stores (W, H, T) directly
        return mat_data['u'], mat_data['v'], is_pycharm

    else:
        raise ValueError(f"Unrecognised file extension '{ext}'. Use .pkl, .pickle, or .mat.")

u_tensors, v_tensors, pycharm_dumb_flag = load_dataset(INPUT_FILE)

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

u = this_is_a_dictionary['training_data'][:, :, 0, :]
v = this_is_a_dictionary['training_data'][:, :, 1, :]

u_training_mean = np.nanmean(u)
u_training_std = np.nanstd(u)
v_training_mean = np.nanmean(v)
v_training_std = np.nanstd(v)

magnitudes = np.sqrt(u ** 2 + v ** 2)
avg_magnitude = np.nanmean(magnitudes)
print("Average vector magnitude in training data:", avg_magnitude)

try:
    path = '../data.pickle' if pycharm_dumb_flag else 'data.pickle'
    with open(path, 'wb') as file:
        pickle.dump([training_data, validation_data, test_data], file)
    print(f"Pickle saved to: {path}")
except Exception as e:
    print("Failed to pickle:", e)

print ("you've been pickle'd")