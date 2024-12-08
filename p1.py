# Program 1: Implement a perceptron model in Python to simulate the behavior of 
#an AND gate and OR gate. Use sigmoid activation function and a single perceptron 

import numpy as np

# #######Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

 ########  # Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x) 


X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [0], [0], [1]])

##### Initialize weights and bias
#np.random.seed(0)  # For reproducibility
weights = np.random.rand(2, 1)
bias = np.random.rand(1)
print("Initial weights:", weights)
print("Initial bias:", bias)


# Training parameters
learning_rate = 0.01
epochs = 10000


# Testing the AND gate
print("Testing the AND gate before training:")
for i in range(len(X)):
    input_data = X[i]
    output = sigmoid(np.dot(input_data, weights) + bias)
    print(f"Input: {input_data}, normarl_output:{output}, Output: {np.round(output)}")


# Training loop
for epoch in range(epochs):
    # Forward pass
    input_layer = X
    weighted_sum = np.dot(input_layer, weights) + bias
    activated_output = sigmoid(weighted_sum)
    print(activated_output)
    # Calculate the error
    error = y - activated_output

    # Backpropagation
    adjustment = error * sigmoid_derivative(activated_output)
    weights += np.dot(input_layer.T, adjustment) * learning_rate
    bias += np.sum(adjustment) * learning_rate




########### Test the trained model
print("Final weights:", weights)
print("Final bias:", bias)

# Testing the AND gate
print("Testing the AND gate after training:")
for i in range(len(X)):
    input_data = X[i]
    output = sigmoid(np.dot(input_data, weights) + bias)
    print(f"Input: {input_data}, Output: {np.round(output)}")
