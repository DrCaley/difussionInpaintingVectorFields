from scipy.io import loadmat
import numpy as np
import pickle
import datetime

mat_data = loadmat("../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat")
u_tensors=mat_data['u']

trainingData = np.zeros([len(u_tensors),len(u_tensors[0]),1])
validationData = np.empty([len(u_tensors),len(u_tensors[0]),1])
testData = np.empty([len(u_tensors),len(u_tensors[0]),1])

for i in range(0,len(u_tensors[0][0]),130):
    trainingData = np.append(trainingData,u_tensors[::,::,i:i+70],2)
    validationData = np.append(validationData, u_tensors[::, ::, i+80:i + 95],2)
    testData = np.append(testData, u_tensors[::, ::, i+105:i + 120],2)

trainingData = trainingData[::,::,1:]
validationData = validationData[::,::,1:]
testData = testData[::,::,1:]

metaData = {'dateMade':datetime.datetime.now(),
            'dataSplitStrat':'Alternating chucks of 120 frames go into training, validation'
                             'and test. The first 70 go into training, space of 10, 15 in'
                             ' validation, space of 10, last 15 go into test, space of 10'}

this_is_a_dictionary = {'trainingData':trainingData,
                        'validationData':validationData,
                        'testData':testData,
                        'metaData':metaData}


with open('data.pickle', 'wb') as file:
    pickle.dump([trainingData,validationData,testData], file)

print ("hello")