#Reference: http://python3.codes/neural-network-python-part-1-sigmoid-function-gradient-descent-backpropagation/
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

#Step1. prepare dataset
Labels = []
Features = []

file_location = './Dataset/cross200.txt'
fp = open(file_location, 'r')
for line in fp.readlines():
    line_elements = line.split()
    Labels.append(int(line_elements.pop()))
    Features.append([float(i) for i in line_elements])

dataset = list(zip(Features, Labels))

#Step2. divide dataset into training and testing set
#random.shuffle(dataset)
training_size = int(len(dataset)) #int(len(dataset)*0.8)
training_set = dataset #dataset[0:training_size]
random.shuffle(training_set)
testing_size = int(len(dataset)) #int(len(dataset)) - training_size
testing_set = dataset[training_size:]

training_features = []
training_labels = []
for sample in training_set:
    #training_features.append([sample[0], sample[1]])
    #training_labels.append([sample[2]-1])
    training_features.append(sample[0])
    training_labels.append([sample[1]-1])

testing_features = []
testing_labels = []
#for sample in testing_set:
for sample in dataset:
    #testing_features.append([sample[0], sample[1]])
    #testing_labels.append([sample[2]-1])
    testing_features.append(sample[0])
    testing_labels.append(sample[1]-1)


def plot_dataset(dataset):
    label = []
    feature1 = []
    feature2 = []
    for data in dataset:
        feature1.append(data[0][0])
        feature2.append(data[0][1])
        label.append(data[1])
    dataset_to_plot = list(zip(feature1, feature2, label))
    #dataframe_obj = pd.read_csv(file_location, header=None, sep=" ", names=['col1', 'col2', 'col3'])
    dataframe_obj = pd.DataFrame(data=dataset_to_plot, columns=['feature1', 'feature2', 'label'])
    dataframe_obj.plot(kind='scatter', x='feature1', y='feature2', c='label')
    plt.show()
    #print dataframe_obj.info()
    #print dataframe_obj.head(10)
    #print dataframe_obj.tail(10)
    #print dataframe_obj['feature1']

#Step3. train a neural network
epochs = 3000           # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 10, 1

#X = np.array([[0,0], [0,1], [1,0], [1,1]])
#Y = np.array([ [0],   [1],   [1],   [0]])
X = np.array(training_features)
Y = np.array(training_labels)
J = []

def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid #return sigmoid(x) * (1 - sigmoid(x))
                                                # weights on layer inputs

def activation(x, type='sigmoid', derivative=False):
	if type == 'sigmoid':
		sigmoid = 1/(1 + np.exp(-x))
		if not derivative:
			return sigmoid
		else:
			return sigmoid * (1 - sigmoid) #s'(x) = s(x) * (1 - s(x))
	elif type == 'tanh':
		pass
	else:
		pass

Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize))

#training
learning_rate = 0.25
alpha = 0.01 #used in momentum
for i in range(epochs):
    prev_Wz = Wz
    prev_Wh = Wh
    #forward propagation
    H = activation(np.dot(X, Wh))                  # hidden layer results
    Z = activation(np.dot(H, Wz))                  # output layer results
    E = Y - Z                                   # how much we missed (error)
    sum_of_squared_error = 0
    for error in E:
        sum_of_squared_error += (error[0]*error[0])

    #backpropagation
    #w(t+1) = w(t) - learning_rate * dJ/dw
	#dZ = E * sigmoid_(Z)                        # delta Z
	#dH = dZ.dot(Wz.T) * sigmoid_(H)             # delta H
    dZ = E * activation(np.dot(H, Wz), derivative=True)                        # delta Z
    dH = dZ.dot(Wz.T) * activation(np.dot(X, Wh), derivative=True)             # delta H
    dZ = H.T.dot(dZ)
    dH = X.T.dot(dH)
    Wz += (learning_rate*dZ + alpha*(Wz - prev_Wz))                          # update output layer weights
    Wh += (learning_rate*dH + alpha*(Wh - prev_Wh))                           # update hidden layer weights
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
#for i in range(testing_size):
for i in range(int(len(dataset))):
    if(test_labels[i] == 1):
        x1.append(testX[i][0])
        y1.append(testX[i][1])
    else:
        x2.append(testX[i][0])
        y2.append(testX[i][1])
plt.scatter(x1, y1, color='r')
plt.scatter(x2, y2, color='g')
plt.show()

plot_dataset(dataset)
