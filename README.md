# NeuralNetwork-for-Binary-Classification

## Dataset: LBW_DATASET

<h3>Implementation:</h3> 
- What are used: Just pandas and numpy, scikit for train_test_split.

1. The Neural network we implemented consists of 4 layers in total of which the 2 hidden layers contain 6 and 3 nodes respectively.

2. We used Sigmoid and its derivative and Relu activation functions.

3. Sigmoid is used for forward propagation and sigmoid_derivative is used in back proagation

4. We used Mean square error as our loss function.

5. We update the weights and bias for the fixed learning rate until the model is well trained

<h3>  Hyperparameters: </h3> 

- Learning Rate : 0.01
- No of epochs: 20000
- Hidden layer 1: Conists 6 nodes
- Hidden layer 2 : Consists 3 nodes

<h3>  Detailed steps to run files: </h3> 

1. PreProcess.py file has the code to implement pre rocessing of data, so initially run that to get preprocessed data which stores it into "PreProcessed_LBW"
2. Read that csv into to get data in file named "NeuralNet.py" in src folder.
3. Run "NeuralNet.py" which trains the model and tests to produce accuracy metrics such as confusion matrix and F1 score along with accuracy.


