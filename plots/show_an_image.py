import os
import sys
from plot_data_tool import plot_vector_field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from data_prep.ocean_image_dataset import OceanImageDataset
from data_prep.data_initializer import DDInitializer


import pickle
with open("C:\\Users\\Matthew\\Documents\\GitHub\\difussionInpaintingVectorFields\\data.pickle", "rb") as f:
    train_np, val_np, test_np = pickle.load(f)
print(train_np.shape)


data_init = DDInitializer()

training_tensor, test_tensor, validation_tensor = data_init.get_tensors()
tensor_to_draw = test_tensor[0]

print(tensor_to_draw.shape)