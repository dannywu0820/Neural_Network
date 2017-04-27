import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

#Step1. prepare dataset
feature1 = []
feature2 = []
label = []

file_location = './Dataset/cross200.txt'
#file_location = './Dataset/elliptic200.txt'
fp = open(file_location, 'r')
for line in fp.readlines():
    #print line[0:6].strip() + ',' + line[7:13].strip() + ',' + line[14]
    feature1.append(float(line[0:6].strip()))
    feature2.append(float(line[7:13].strip()))
    label.append(int(line[14]))

dataset = list(zip(feature1, feature2, label))

#Step2. divide dataset into training and testing set
random.shuffle(dataset)
training_size = int(len(dataset)*0.8)
testing_size = int(len(dataset)) - training_size
training_set = dataset[0:training_size]
testing_set = dataset[training_size:]

training_features = []
training_labels = []
for sample in training_set:
    training_features.append([sample[0], sample[1]])
    training_labels.append([sample[2]-1])
testing_features = []
testing_labels = []
for sample in testing_set:
    testing_features.append([sample[0], sample[1]])
    testing_labels.append([sample[2]-1])



#dataframe_obj = pd.read_csv(file_location, header=None, sep=" ", names=['col1', 'col2', 'col3'])
dataframe_obj = pd.DataFrame(data=dataset, columns=['feature1', 'feature2', 'label'])
#print dataframe_obj.info()
#print dataframe_obj.head(10)
#print dataframe_obj.tail(10)
#print dataframe_obj['feature1']
dataframe_obj.plot(kind='scatter', x='feature1', y='feature2', c='label')
plt.show()

#Step3. train a neural network
epochs = 1000           # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 5, 1

#X = np.array([[0,0], [0,1], [1,0], [1,1]])
#Y = np.array([ [0],   [1],   [1],   [0]])
X = np.array(training_features)
Y = np.array(training_labels)
J = []

def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid
                                                # weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize))

#training
learning_rate = 1
alpha = 1
for i in range(epochs):

    H = sigmoid(np.dot(X, Wh))                  # hidden layer results
    Z = sigmoid(np.dot(H, Wz))                  # output layer results
    E = Y - Z                                   # how much we missed (error)
    sum_of_squared_error = 0
    for error in E:
        sum_of_squared_error += (error[0]*error[0])

    dZ = E * sigmoid_(Z)                        # delta Z
    dH = dZ.dot(Wz.T) * sigmoid_(H)             # delta H
    Wz +=  learning_rate*H.T.dot(dZ)                          # update output layer weights
    Wh +=  learning_rate*X.T.dot(dH)                          # update hidden layer weights
    J.append(sum_of_squared_error)
print sum_of_squared_error/training_size

#print(Z)
#testing
testX = np.array(testing_features)
testY = np.array(testing_labels)

H = sigmoid(np.dot(testX, Wh))                  # hidden layer results
Z = sigmoid(np.dot(H, Wz))                  # output layer results
E = testY - Z
for error in E:
    sum_of_squared_error += (error[0]*error[0])
print sum_of_squared_error/testing_size

x = list(range(epochs))
plt.xlabel('Epoch')
plt.ylabel('Sum of Squared Error')
plt.plot(x, J, 'b--')
plt.show()

test_labels = []
for result in Z:
    if(result[0] > 0.5):
        test_labels.append(1)
    else:
        test_labels.append(0)
print test_labels
x1 = []
y1 = []
x2 = []
y2 = []
for i in range(testing_size):
    if(test_labels[i] == 1):
        x1.append(testX[i][0])
        y1.append(testX[i][1])
    else:
        x2.append(testX[i][0])
        y2.append(testX[i][1])
plt.scatter(x1, y1, color='r')
plt.scatter(x2, y2, color='g')
plt.show()
