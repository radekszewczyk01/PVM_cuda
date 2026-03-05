# GPU Multi-Layer Perceptron (gpu_mlp.py) Documentation

## Overview

The `gpu_mlp.py` module implements multi-layer perceptron (MLP) models optimized for execution on GPU. This allows for efficient training and inference of neural networks, leveraging the parallel processing capabilities of modern GPUs.

## Key Components

### Classes

#### MLP
- **Description**: The main class representing the multi-layer perceptron model.
- **Responsibilities**:
  - Initialize the network architecture (number of layers, neurons per layer).
  - Define the forward pass for input data.
  - Implement the backpropagation algorithm for training the model.

### Functions

#### `__init__(self, layer_sizes)`
- **Parameters**: 
  - `layer_sizes`: A list of integers representing the number of neurons in each layer.
- **Description**: Initializes the MLP with the specified architecture. Sets up weights and biases for each layer.

#### `forward(self, X)`
- **Parameters**: 
  - `X`: Input data for the forward pass.
- **Returns**: Output of the network after applying the activation functions.
- **Description**: Computes the output of the network by passing the input through each layer.

#### `backward(self, X, y, learning_rate)`
- **Parameters**: 
  - `X`: Input data.
  - `y`: True labels for the input data.
  - `learning_rate`: The rate at which the model updates weights during training.
- **Returns**: None.
- **Description**: Performs backpropagation to update the weights and biases based on the error between predicted and true labels.

#### `train(self, X, y, epochs, batch_size)`
- **Parameters**: 
  - `X`: Training data.
  - `y`: Corresponding labels.
  - `epochs`: Number of training iterations.
  - `batch_size`: Size of mini-batches for training.
- **Returns**: None.
- **Description**: Trains the MLP model using the provided training data over a specified number of epochs.

#### `predict(self, X)`
- **Parameters**: 
  - `X`: Input data for which predictions are to be made.
- **Returns**: Predicted labels for the input data.
- **Description**: Uses the trained model to make predictions on new data.

## Key Calculations

- **Forward Pass**: The output of each layer is calculated using the formula:
  - `output = activation_function(weights * input + bias)`
  
- **Backpropagation**: The weight updates are calculated using the gradient of the loss function with respect to the weights:
  - `weights -= learning_rate * gradient`

## Usage

To use the MLP class, instantiate it with the desired layer sizes, and then call the `train` method with your training data. After training, you can use the `predict` method to make predictions on new data.

```python
from gpu_mlp import MLP

# Example usage
model = MLP(layer_sizes=[784, 128, 10])  # Example for MNIST dataset
model.train(X_train, y_train, epochs=10, batch_size=32)
predictions = model.predict(X_test)
```

This module is designed to be efficient and scalable, making it suitable for large datasets and complex neural network architectures.