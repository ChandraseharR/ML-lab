import numpy as np
import math

class Perceptron:
    def __init__(self):
        self.weights = []

    # activation function
    def activation(self, data):
        activation_val1 = np.dot(data,self.weights)
        activation_val=1/(1+math.exp(activation_val1))
        print(activation_val)
        return 1 if activation_val >= 0.5 else 0

    def fit(self, X, y, lrate, epochs):
        # initializing weight vector
        self.weights = [0.0 for i in range(len(X.columns))]
        # no.of iterations to train the neural network
        for epoch in range(epochs):
            print(str(epoch + 1), "epoch has started...")
            for index in range(len(X)):
                x = X.iloc[index]
                if epoch==0:
                    print(x)
                predicted = self.activation(x)
                print(predicted)
                # check for misclassification
                if y.iloc[index]=='Iris-virgincica':
                    val=1
                else:
                    val=0
                if (val== predicted):
                    pass
                else:
                    # calculate the error value
                    print(y.iloc[index])
                    print(predicted)
                    error = val - predicted

                    # updation of associated self.weights acccording to Perceptron training rule
                    for j in range(len(x)):
                        self.weights[j] = self.weights[j] + lrate * error * x[j]

    # training perceptron for the given data
    def predict(self, x_test):
        predicted = []
        for i in range(len(x_test)):
            # prediction for test set using obtained weights
            predicted.append(self.activation(x_test.iloc[i]))
        return predicted

    def accuracy(self, predicted, original):
        correct = 0
        lent = len(predicted)
        for i in range(lent):
            if original.iloc[i] == 'Iris-virgincica':
                val = 1
            else:
                val = 0
            if (predicted[i] ==val ):
                correct += 1
        return (correct / lent) * 100

    def getweights(self):
        return self.weights


import pandas as pd
from sklearn.model_selection import train_test_split
#read data from .csv file
data=pd.read_csv("iris.csv")
dada=data[0:151]
print(data)
data.columns=["petal_length","petal_width","sepal_length","sepal_width","class"]
classes=data["class"]
data=data.drop(columns="class")
#splitting test and train data for iris
x_train,x_test,y_train,y_test=train_test_split(data,classes)
model=Perceptron()
model.fit(x_train,y_train,0.5,10)
pred=model.predict(x_test)
print("accuracy: ",model.accuracy(pred,y_test))
print("weights: ",model.getweights())
