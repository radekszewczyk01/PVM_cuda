from gpu_mlp import MLP

# Initialize the model
model = MLP(input_size=784, hidden_layers=[128, 64], output_size=10)

# Train the model
model.train(training_data, training_labels)

# Make predictions
predictions = model.infer(test_data)