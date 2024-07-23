"""
Explanation

Initialize Parameters: Set the initial weights (ww) and bias (bb) to zero.
Compute Cost: Calculate the cost using the logistic regression cost function.
Gradient Descent: Update the weights and bias iteratively using the gradient descent update rules.
Sigmoid Function: Use the sigmoid function to predict probabilities.
Training: Train the model using the example dataset and labels.

By running this code, you will get the final weights and bias after training, which can then be used to make predictions on new data.
"""

import numpy as np

# Sigmoid function
def sigmoid(z):
    """
    In logistic regression,
    the prediction is made using the sigmoid function applied to the linear combination of the input features:
    where:
    z=wTx+b
    """
    # return value [0..1] --> binary categories
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, w, b):
    """
    To measure how well the model is performing,
    we use a cost function (also called loss function).
    For logistic regression, the cost function is the log loss (also known as binary cross-entropy loss):

    Where:
    * m is the number of training examples.
    * y(i) is the true label for the i-th training example.
    * hw​(x(i)) is the predicted probability for the i-th training example, which is calculated as σ(z)σ(z).

    """
    m = X.shape[0]
    cost = -1/m * np.sum(y * np.log(sigmoid(np.dot(X, w) + b)) + (1 - y) * np.log(1 - sigmoid(np.dot(X, w) + b)))
    return cost

# Gradient descent
def gradient_descent(X, y, w, b, alpha, num_iterations):
    """
    The partial derivatives of the cost function are calculated as follows:
    """
    m = X.shape[0]
    for i in range(num_iterations):
        z = np.dot(X, w) + b
        predictions = sigmoid(z)
        
        dw = 1/m * np.dot(X.T, (predictions - y))
        db = 1/m * np.sum(predictions - y)
        
        w = w - alpha * dw
        b = b - alpha * db
        
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {compute_cost(X, y, w, b)}")
    
    return w, b

# Example data
X = np.array([
    [1.2, 0.7, 3.1],
    [2.3, 1.4, 0.6],
    [0.8, 2.5, 1.3],
    [1.5, 0.3, 2.4],
    [2.1, 1.8, 0.9]
])
y = np.array([1, 0, 1, 1, 0])  # Example labels

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0

# Set learning rate and number of iterations
alpha = 0.01
num_iterations = 1000

# Perform gradient descent
w, b = gradient_descent(X, y, w, b, alpha, num_iterations)

# Final weights and bias
print("Final weights:", w)
print("Final bias:", b)

