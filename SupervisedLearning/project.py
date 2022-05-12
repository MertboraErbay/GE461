from binascii import b2a_base64
import os
import numpy as np
import matplotlib.pyplot as plt
from random import random

directory = os.path.dirname(__file__)

trainF = open(directory + "/data_supervised_learning_project/train1.txt",'r').readlines()
testF = open(directory + "/data_supervised_learning_project/test1.txt",'r').readlines()

train = []
test = []

for row in trainF:
    temp = []
    thisRow = row.replace('\n',"").split('\t')
    for item in thisRow:
        value = float(item)
        temp.append(value)
    train.append(temp)    

for row in testF:
    temp = []
    thisRow = row.replace('\n',"").split('\t')
    for item in thisRow:
        value = float(item)
        temp.append(value)
    test.append(temp)  

test_array = np.array(test)
train_array = np.array(train)

def normalize(data):
    normalized = []
    for item in data:
        normalized.append((item - np.mean(data))/np.std(data))
    return normalized

train_input = normalize(train_array[:,0])
train_output = train_array[:,1]
test_input = normalize(train_array[:,0])
test_output = train_array[:,1]

def init_weights(nx, nh, ny):
    np.random.seed(65)
    w1 = np.random.rand(nx, nh)
    w2 = np.random.rand(nh, ny)
    return w1, w2

def init_bias(nx,nh,ny):
    np.random.seed(34)
    b1 = np.random.rand(nx, nh)
    b2 = np.random.rand(nx, ny)
    return b1,b2

def no_hidden_ann(x,y,epochs,lr):
    np.random.seed(60)
    weights = np.random.rand(1,2)
    sse = []
    for epoch in range(0,epochs):
        pred = weights[0][0] + np.dot(weights[0][1],x)
        dw0 = -2*sum(y-pred)
        dw1 = -2*sum(x*(y-pred))
        weights[0][0] -= lr*dw0
        weights[0][1] -= lr*dw1
        sse.append(sum((y-pred)**2))
        print("SSE at epoch " + str(epoch) + ": " + str(sum((y-pred)**2)))
    plt.plot(sse)
    plt.title("SSE per epoch")
    plt.show()    
    return weights,pred
    
#weights, pred = no_hidden_ann(train_array[:,0],train_array[:,1],25000,0.000001) #SSE = 71387.6546
#plt.scatter(train_array[:,0], train_array[:,1]) 
#plt.plot(train_array[:,0], pred, color='orange')
#plt.title("Regression on Train")
#plt.show()

#nweights,npred = no_hidden_ann(train_input,train_output,25000,0.00001) #SSE 

#plt.scatter(train_input, train_output) 
#plt.plot(train_input, npred, color='orange')
#plt.title("Regression on Train Normalized")
#plt.show()

class nHiddenANN:
    def __init__(self, nh):
        self.nh = nh
        self.weights1 , self.weights2 = init_weights(1, nh, 1)
        self.bias1,self.bias2 = init_bias(1,nh,1)

    def sigmoid(self, x):
        return  1 / (1 + np.exp(-x))

    def backsigmoid(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def forward(self,x): 
        w1 = self.weights1
        w2 = self.weights2
        b1 = self.bias1
        b2 = self.bias2
        A1 = self.sigmoid(np.dot(x,w1)+b1)
        output = self.sigmoid(np.dot(A1,w2)+b2)
        return output, A1
    
    def backward (self, x ,y, output, A1, lr):
        b1 = self.bias1
        b2 = self.bias2
        w1 = self.weights1
        w2 = self.weights2
        dw2 = np.dot(A1.T, 2*(y - output))
        dw1 = np.dot(x.T,(np.dot(2*(y - output), w2.T)* self.backsigmoid(np.dot(x,w1)+b1)))
        db2 = 2*(y - output)
        db1 = 2*(y - output)* w2.T * self.backsigmoid(np.dot(x,w1) + b1)
        w2 += lr * dw2
        w1 += lr * dw1
        b1 += lr * sum(db1)
        b2 += lr * sum(db2)
        self.bias1 = b1
        self.bias2 = b2
        self.weights2 = w2
        self.weights1 = w1
        return w1,w2, b1, b2

    def train(self,xlist, y, e, lr):
        w1 = self.weights1
        w2 = self.weights2
        b1 = self.bias1
        b2 = self.bias2
        x = np.array(xlist).reshape(60,1)
        y = y.reshape(60,1)
        for i in range(0,e):
            output, A1 = self.forward(x)
            self.weights1,self.weights2,self.bias1,self.bias2 =  self.backward(x,y, output, A1, lr)
            pred,_ =  self.forward(x)
            print("epoch: "+ str(i) + " SSE: " + str(sum((y-pred)**2)) )
        return w1,w2,b1,b2,pred



#for n in [2,4,8,16,32]:
model = nHiddenANN(1)
e=10000
lr = 0.000001
w1,w2,b1,b2,pred = model.train(train_input, train_output, e = e, lr = lr)

plt.scatter(train_input, train_output) 
plt.scatter(train_input, pred, color='orange')
plt.title("ANN " +str(1) +" hidden layers epoch = " + str(e) + ", lr = " + str(lr) + " Normalized")
plt.show()