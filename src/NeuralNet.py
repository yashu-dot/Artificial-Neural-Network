

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df  =  pd.read_csv('PreProcessed_LBW.csv') #Read the pre processed data 

X = df.drop(columns=['Result'],axis=1).reset_index(drop=True) #Extracting the result column 
Y = df['Result']


#---------Performing train-test split for training and testing the Neural Network using sklearn's model_selection---------#
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state= 0)
X_train = X_train.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)

#----------------------Class definition of the Artificial Neural Network------------------#
class NN:
    def __init__(self,X,Y,X_test,Y_test,h1=6,h2=3,learning_rate=0.01,epochs=100): #Defining the parameters of the neural network

      '''Parameters:
      1. Number of hidden layers -> 2
      2. Number of nodes in hidden layers -> [6,3]
      3. Learning rate -> 0.01
      4. Number of epochs -> 20000
      5. X has the input data
      6. Y has our desired output which we use to train our model
      7. X_test and Y_test holds the testing data 
      6. h1 -> number of neurons in 1st hidden layer
      7. h2 -> number of neurons in 2nd hidden layer
      '''
      self.X=X
      self.Y=Y[:,None]
      self.X_test = X_test
      self.Y_test = Y_test
      self.epochs = 20000

      np.random.seed(2)
      self.input_nodes = X.shape[1]   # number of features in the training data

      self.h1 = h1
      self.h2 = h2
      self.output_nodes = self.Y.shape[1]
      self.learning_rate = learning_rate


#----------------Initialising the weights and bias at random for the Neural network----------------#
      '''
      w1 -> Weights of layer 1, of shape [#InputNodes,#hidden layer1]
      w2 -> weights of layer 2, of shape [#hidden layer1,#hidden layer2]
      w3 -> weights of layer 3, of shape [#hidden layer2,#outputNodes]
      b1 -> Bias for layer1
      b2 -> Bias for layer2
      b3 -> Bias for layer3
      '''
      self.w1 = 2 * np.random.random((self.input_nodes,self.h1))-1
      self.b1 = 2 * np.random.random([1,self.h1]) - 1
      self.w2 = 2 * np.random.random((self.h1,self.h2))-1
      self.b2 = 2 * np.random.random([1,self.h2]) - 1
      self.w3 = 2 * np.random.random((self.h2,self.output_nodes))-1
      self.b3 = 2 * np.random.random([1,self.output_nodes]) - 1

      self.fit(self.X,self.Y)
      self.predict(self.X_test)

#----------------Activation function definitions(sigmoid,relu and their derivatives) and function to caluclate M.S.E---------------#
    def sigmoid(self,Z): 
      return 1.0/(1.0+np.exp(-Z))

    def sigmoid_prime(self,Z):
      return Z * (1-Z)
    
    def relu(self,x):
      return np.maximum(0,x)

    def MSE(self,x,y):
      return np.average((x-y)**2)

#--------------Training function which perfroms forward and back propagation, updating the weights and bias, thus training out ANN---------#
    def fit(self,X,Y):
      loss = 0

      #Forward Propagation
      for i in range(self.epochs):
        c1 = self.sigmoid((np.dot(X,self.w1) + self.b1)) 
        c1 = self.relu(c1)
        c2 = self.sigmoid(((np.dot(c1,self.w2) + self.b2)))
        c2 = self.relu(c2)
        c3 = self.sigmoid((np.dot(c2,self.w3)+self.b3))

        error = self.Y - c3
      
      #Back Propagation
        c3_d = error * self.sigmoid_prime(c3)
        c2_d = c3_d.dot(self.w3.T) * self.sigmoid_prime(c2)
        c1_d = c2_d.dot(self.w2.T) * self.sigmoid_prime(c1)

        self.w3 = np.add(self.w3,c2.T.dot(c3_d) * self.learning_rate)
        self.b3 = np.add(self.b3,np.sum(c3_d,axis=0) * self.learning_rate)
        self.w2 = np.add(self.w2, c1.T.dot(c2_d) * self.learning_rate)
        self.b2 += np.sum(c2_d,axis=0) * self.learning_rate      
        self.w1 = np.add(self.w1, X.T.dot(c1_d) * self.learning_rate)
        self.b1 += np.sum(c1_d,axis=0) * self.learning_rate     

        loss = self.MSE(Y,c3)
        if i%1000 == 0:
          print('loss at :',i, 'is' , loss/len(X))
    
#-----------Function to predict classes of input data--------------------#
    def predict(self,X):
    
      
      c1 = self.sigmoid((np.dot(X,self.w1) + self.b1)) 
      c1 = self.relu(c1)
      c2 = self.sigmoid(((np.dot(c1,self.w2) + self.b2)))
      c2 = self.relu(c2)
      c3 = self.sigmoid((np.dot(c2,self.w3)+self.b3))

      return c3

#-----------Function to caluclate accuracy metrics, printing Confusion Matrix and Accuracy----------------#
    def CM(self,y_test,Y_test_obs):
    
      for i in range(len(Y_test_obs)):
        if(Y_test_obs[i]>0.6):
          Y_test_obs[i]=1
        else:
          Y_test_obs[i]=0
        
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
    
      for i in range(len(Y_test)):
        if(Y_test[i]==1 and Y_test_obs[i]==1):
          tp=tp+1
        if(Y_test[i]==0 and Y_test_obs[i]==0):
          tn=tn+1
        if(Y_test[i]==1 and Y_test_obs[i]==0):
          fp=fp+1
        if(Y_test[i]==0 and Y_test_obs[i]==1):
          fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

      p= tp/(tp+fp)
      r=tp/(tp+fn)
      f1=(2*p*r)/(p+r)
        
      print("Confusion Matrix : ")
      print(cm)
      print("\n")
      print(f"Precision : {p}")
      print(f"Recall : {r}")
      print(f"F1 SCORE : {f1}")
      acc = float((tp+tn)/(tp+tn+fp+fn))
      acc = acc*100
      print("accuracy reached is -> %.3f%%" %(acc))

nn = NN(X_train,Y_train,X_test,Y_test) #Creating Class object and training our Neural network with our data
nn.CM(Y_test,nn.predict(X_test))

