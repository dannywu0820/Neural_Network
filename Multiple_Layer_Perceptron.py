#Reference: http://python3.codes/neural-network-python-part-1-sigmoid-function-gradient-descent-backpropagation/
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np

def read_dataset(file_location):
    labels = []
    features = []

    fp = open(file_location, 'r')
    for line in fp.readlines():
        line_elements = line.split()
        labels.append(int(line_elements.pop()))
        features.append([float(i) for i in line_elements])

    dataset = list(zip(features, labels))
    random.shuffle(dataset)

    return dataset

def plot_dataset(dataset):
    label = []
    feature1 = []
    feature2 = []
    for data in dataset:
        feature1.append(data[0][0])
        feature2.append(data[0][1])
        label.append(data[1])
    data_to_plot = list(zip(feature1, feature2, label))
    dataframe_obj = pd.DataFrame(data=data_to_plot, columns=['Feature1', 'Feature2', 'Label'])

    dataframe_obj.plot(kind='scatter', x='Feature1', y='Feature2', c='Label')
    #dataframe_obj = pd.read_csv(file_location, header=None, sep=" ", names=['col1', 'col2', 'col3'])
    #print dataframe_obj.info()
    #print dataframe_obj.head(10)
    #print dataframe_obj.tail(10)

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

def build_model(training_set, structure, epoch=10, rho=0.1, alpha=0.1): #0 <= alpha < 1
    #inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 8, 1
    inputLayerSize, hiddenLayerSize, outputLayerSize = structure['Input'], structure['Hidden'], structure['Output']
    training_features = []
    training_labels = []
    for sample in training_set:
        training_features.append(sample[0])
        training_labels.append([sample[1]-1])
    X = np.array(training_features) #X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array(training_labels)   #Y = np.array([ [0],   [1],   [1],   [0]])
    J = []

    Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
    Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize))

    for i in range(epoch):
        prev_Wz = Wz
        prev_Wh = Wh
        #forward propagation
        H = activation(np.dot(X, Wh)) # hidden layer results
        Z = activation(np.dot(H, Wz)) # output layer results
        E = Y - Z                     # how much we missed (error)

        #backpropagation
        #w(t+1) = w(t) - (rho * dJ/dw) + (alpha * [w(t)-w(t-1)])
        dZ = E * activation(np.dot(H, Wz), derivative=True)                        #delta Z
        dH = dZ.dot(Wz.T) * activation(np.dot(X, Wh), derivative=True)             #delta H
        dZ = H.T.dot(dZ)
        dH = X.T.dot(dH)
        Wz += ((rho * dZ) + alpha * (Wz - prev_Wz)) #update output layer weights
        Wh += ((rho * dH) + alpha * (Wh - prev_Wh)) #update hidden layer weights

        sum_of_squared_error = 0
        for error in E:
            sum_of_squared_error += (error[0]*error[0])                          #update hidden layer weights
        J.append(sum_of_squared_error)

    x = list(range(epoch))
    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Sum of Squared Error')
    plt.plot(x, J, 'b--')

    model = {'Wz': Wz, 'Wh': Wh, 'Learning_Curve': J}
    return model

def validate_model(validation_set, model):
    validation_features = []
    validation_labels = []
    for sample in validation_set:
        validation_features.append(sample[0])
        validation_labels.append(sample[1]-1)

    X = np.array(validation_features) #X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array(validation_labels)   #Y = np.array([ [0],   [1],   [1],   [0]])

    H = activation(np.dot(X, model['Wh']))
    Z = activation(np.dot(H, model['Wz']))
    E = Y - Z

    sum_of_squared_error = 0
    for error in E:
        sum_of_squared_error += (error[0]*error[0])
    model['ValidationError'] = sum_of_squared_error

def cross_validation(training_data, k=4): #k-fold cross validation
    fold_size = int(len(training_data)/k)
    training_folds = []

    for i in range(0, k):
        new_fold = []
        for j in range(fold_size*i, fold_size*(i+1)):
            new_fold.append(training_data[j])
        training_folds.append(new_fold)

    errors = []
    min_error = sys.maxint
    best_model = None
    for i in range(0, k):
        validation_set = training_folds[0]
        training_set = training_folds[1] + training_folds[2] + training_folds[3]

        #train a model
        structure = {'Input': 2, 'Hidden': 8, 'Output': 1}
        epoch, learning_rate, momentum_rate = 10000, 0.1, 0
        model = build_model(training_set, structure, epoch, learning_rate, momentum_rate)
        validate_model(validation_set, model)
        errors.append(model['ValidationError'])
        if(model['ValidationError'] < min_error):
            min_error = model['ValidationError']
            best_model = model

        first_fold = training_folds.pop(0)
        training_folds.append(first_fold)

    print 'Validation Errors for Hidden Layer Size = ' + str(structure['Hidden'])
    print errors

    return best_model

def testing(testing_set, model):
    testing_features = []
    testing_labels = []
    for sample in testing_set:
        testing_features.append(sample[0])
        testing_labels.append(sample[1]-1)

    X = np.array(testing_features) #X = np.array([[0,0], [0,1], [1,0], [1,1]])
    target_Y = np.array(testing_labels)   #Y = np.array([ [0],   [1],   [1],   [0]])

    H = activation(np.dot(X, model['Wh']))
    Z = activation(np.dot(H, model['Wz']))

    estimated_Y = []
    for result in Z:
        #print result
        if(result[0] > 0.5):
            estimated_Y.append(1)
        else:
            estimated_Y.append(0)

    class1 = {'x': [], 'y': []}
    class2 = {'x': [], 'y': []}
    correctness = 0
    for i in range(int(len(testing_set))):
        if(estimated_Y[i] == target_Y[i]):
            correctness+=1

        if(estimated_Y[i] == 1):
            class1['x'].append(X[i][0])
            class1['y'].append(X[i][1])
        else:
            class2['x'].append(X[i][0])
            class2['y'].append(X[i][1])
    print 'Correctness: ' + str(float(correctness)*100/float(len(testing_set))) + '%'

    plt.figure(3)
    plt.scatter(class1['x'], class1['y'], color='r')
    plt.scatter(class2['x'], class2['y'], color='b')
    plt.show()

if __name__ == '__main__':
    dataset = read_dataset('./Dataset/cross200.txt')
    #print dataset

    training_size = int(len(dataset) * 0.8)
    testing_size = int(len(dataset) - training_size)
    training_data = dataset #dataset[0:training_size]
    testing_set = dataset #dataset[training_size:]

    #for model selection
    best_model = cross_validation(training_data)

    plot_dataset(dataset)
    #for optimizing model parameters, should retrain the model
    testing(testing_set, best_model)

    plt.show()
