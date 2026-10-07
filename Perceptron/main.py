import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.zeros(X.shape[1])
bias = 0.0

# Hyperparameters
learning_rate = 0.1
epochs = 10

for epoch in range(epochs):

    print("\n" + "-" * 50)
    print(f"Epoch {epoch + 1}")
    print("-" * 50)

    print(f"\nInitial Weights: {weights}")
    print(f"Initial Bias: {bias}")

    errors = 0

    # Go through each training example
    for i in range(len(X)):

        inputs = X[i]
        actual_outputs = y[i]

        # Calculate weighted sum
        weighted_sum = np.dot(inputs, weights) + bias

        # Make prediction
        if weighted_sum >= 0:
            prediction = 1
        else:
            prediction = 0

        # Calculate error
        loss = actual_outputs - prediction

        if loss != 0:
            errors += 1

        # Update weights and bias
        weights = weights + learning_rate * loss * inputs
        bias = bias + learning_rate * loss

        # Print training example details
        print(f"\nInput: {inputs}")
        print(f"Actual output: {actual_outputs}")
        print(f"Prediction: {prediction}")
        print(f"Error: {loss}")
        print(f"Updated Weights: {weights}")
        print(f"Updated Bias: {bias:.2f}")

    # Print errors for this epoch
    print(f"\nErrors in epoch: {errors}")

    # Stop training if there are no errors
    if errors == 0:
        print("\nTraining is successful")
        break


# Test the trained perceptron

print("\n")
print("-" * 50)
print("FINAL MODEL")
print("-" * 50)

print(f"Weights: {weights}")
print(f"Bias: {bias:.2f}")

for i in range(len(X)):

    inputs = X[i]
    actual_outputs = y[i]

    weighted_sum = np.dot(inputs, weights) + bias

    if weighted_sum >= 0:
        prediction = 1
    else:
        prediction = 0

    print(
        f"Input: {inputs}",
        f"Expected: {actual_outputs}",
        f"Predicted: {prediction}"
    )