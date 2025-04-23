from scipy.io import loadmat
import numpy as np
import pickle
import datetime

mat_data = loadmat("../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat")
u_tensors=mat_data['u']
v_tensors=mat_data['v']

u_tensors = np.expand_dims(u_tensors,axis=2)
v_tensors = np.expand_dims(v_tensors,axis=2)

all_tensors = np.append(u_tensors,v_tensors,2)



trainingData = np.zeros([len(all_tensors),len(all_tensors[0]),len(all_tensors[0][0]),1])
validationData = np.zeros([len(all_tensors),len(all_tensors[0]),len(all_tensors[0][0]),1])
testData = np.zeros([len(all_tensors),len(all_tensors[0]),len(all_tensors[0][0]),1])

for i in range(0,len(all_tensors[0][0][0]),130):
    trainingData = np.append(trainingData,all_tensors[::,::,::,i:i+70],3)
    validationData = np.append(validationData, all_tensors[::, ::,::, i+80:i + 95],3)
    testData = np.append(testData, all_tensors[::, ::,::, i+105:i + 120],3)

trainingData = trainingData[::,::,::,1:]
validationData = validationData[::,::,::,1:]
testData = testData[::,::,::,1:]

metaData = {'dateMade':datetime.datetime.now(),
            'dataSplitStrat':'Alternating chucks of 120 frames go into training, validation'
                             'and test. The first 70 go into training, space of 10, 15 in'
                             ' validation, space of 10, last 15 go into test, space of 10',
            'DimensionData':'lat,long,<u><v>,time'}

this_is_a_dictionary = {'trainingData':trainingData,
                        'validationData':validationData,
                        'testData':testData,
                        'metaData':metaData}

#This isn't right. fix it next time
u_training_mean = np.nanmean(this_is_a_dictionary['trainingData'][::,::,0,::])
u_training_std = np.nanstd(this_is_a_dictionary['trainingData'][::,::,0,::])

v_training_mean = np.nanmean(this_is_a_dictionary['trainingData'][::,::,1,::])
v_training_std = np.nanstd(this_is_a_dictionary['trainingData'][::,::,1,::])



with open('../data.pickle', 'wb') as file:
    pickle.dump([trainingData,validationData,testData], file)

print ("hello")