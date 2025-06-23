import numpy as np
import pickle
import datetime



def generate_constant_field(shape, u_val=1.0, v_val=1.0):
    """Generate constant vector field of <u_val, v_val>."""
    u = np.ones(shape) * u_val
    v = np.ones(shape) * v_val
    return u, v

def generate_circular_field(H, W, T, scale=1.0):
    """
    Generate a circular (rotational) vector field at each time step.
    Vector at (x, y) is perpendicular to radius vector from center.
    """
    Y, X = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
    radius = np.sqrt(X**2 + Y**2) + 1e-6  # avoid div-by-zero
    u = -Y / radius * scale
    v = X / radius * scale

    # Repeat the 2D field across time
    u_field = np.stack([u] * T, axis=-1)  # (H, W, T)
    v_field = np.stack([v] * T, axis=-1)
    return u_field, v_field

# Choose type of vector field
field_type = 'constant'  # or 'constant'

# Generate chosen field
if field_type == 'constant':
    u_tensors, v_tensors = generate_constant_field((94, 44, 17040), u_val=1.0, v_val=1.0)
elif field_type == 'circular':
    u_tensors, v_tensors = generate_circular_field(94, 44, 17040, scale=1.0)
else:
    raise ValueError(f"Unknown field type: {field_type}")

# Combine channels
u_tensors = np.expand_dims(u_tensors, axis=2)
v_tensors = np.expand_dims(v_tensors, axis=2)

all_tensors = np.append(u_tensors, v_tensors, axis=2)

# Initialize splits
training_data = np.zeros((94, 44, 2, 1))
validation_data = np.zeros((94, 44, 2, 1))
test_data = np.zeros((94, 44, 2, 1))

# Same split strategy: 120-frame chunks
for i in range(0, 17040, 130):
    training_data = np.append(training_data, all_tensors[:, :, :, i:i+70], axis=3)
    validation_data = np.append(validation_data, all_tensors[:, :, :, i+80:i+95], axis=3)
    test_data = np.append(test_data, all_tensors[:, :, :, i+105:i+120], axis=3)

# Remove dummy first frame
training_data = training_data[:, :, :, 1:]
validation_data = validation_data[:, :, :, 1:]
test_data = test_data[:, :, :, 1:]

# Metadata
meta_data = {
    'date_made': datetime.datetime.now(),
    'data_split_strat': 'Alternating chunks of 120 frames go into training, validation, and test. '
                        'The first 70 go into training, space of 10, 15 in validation, space of 10, '
                        'last 15 go into test, space of 10',
    'dimension_data': 'lat,long,<u><v>,time'
}

# Bundle
this_is_a_dictionary = {
    'training_data': training_data,
    'validation_data': validation_data,
    'test_data': test_data,
    'meta_data': meta_data
}

# Compute u/v stats
u_values = training_data[:, :, 0, :].reshape(-1)
v_values = training_data[:, :, 1, :].reshape(-1)

u_training_mean = np.nanmean(u_values)
u_training_std = np.nanstd(u_values)
v_training_mean = np.nanmean(v_values)
v_training_std = np.nanstd(v_values)

print(f"u mean: {u_training_mean}, u std: {u_training_std}")
print(f"v mean: {v_training_mean}, v std: {v_training_std}")

# Save to pickle
try:
    with open("fake_data.pickle", 'wb') as file:
        pickle.dump([training_data, validation_data, test_data], file)
    print("Fake constant data saved to: fake_data.pickle")
except Exception as e:
    print("Failed to pickle:", e)

print("you've been fake-pickle'd")