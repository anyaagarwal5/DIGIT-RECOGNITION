import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.math import confusion_matrix

# Set the random seed for reproducibility
tf.random.set_seed(3)

# Load the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(f"Type of X_train: {type(X_train)}")

# Print the shapes of the arrays
print(f"Shapes of the datasets: {X_train.shape, Y_train.shape, X_test.shape, Y_test.shape}")

# Visualize the first image in the training set
plt.imshow(X_train[0], cmap='gray')
plt.show()

# Print the label for the first image
print(f"Label of the first image: {Y_train[0]}")
print(f"Type of Y_train: {type(Y_train)}")

# Print unique values in the label arrays
print(f"Unique values in Y_test: {np.unique(Y_test)}")
print(f"Unique values in Y_train: {np.unique(Y_train)}")

# Scale the pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(50, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=10)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")

# Visualize the first image in the test set
plt.imshow(X_test[0], cmap='gray')
plt.show()

# Print the label for the first test image
print(f"Label of the first test image: {Y_test[0]}")

# Predict the labels for the test set
Y_predictions = model.predict(X_test)
print(f"Shape of predictions: {Y_predictions.shape}")

# Get the label with the highest probability
label_prediction = np.argmax(Y_predictions[0])
print(f"Predicted label for the first test image: {label_prediction}")

# Get the predicted labels for all test images
label_pred_for_input_features = [np.argmax(i) for i in Y_predictions]
print(f"Predicted labels for all test images: {label_pred_for_input_features[:10]}")

# Create a confusion matrix
conf_mat = confusion_matrix(Y_test, label_pred_for_input_features)
print(f"Confusion matrix:\n{conf_mat}")

# Save images as PNG files
save_dir = "mnist_png"
os.makedirs(save_dir, exist_ok=True)
for i in range(len(X_train)):
    image = X_train[i]
    label = Y_train[i]

    # Save the image as a PNG file
    file_name = os.path.join(save_dir, f"mnist_{i}_label_{label}.png")
    plt.imsave(file_name, image, cmap='gray')

print("PNG images saved successfully.")

# Predict the label for a new image
def predict_image(image_path):
    # Read and preprocess the image
    image_to_be_predicted = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_to_be_predicted is None:
        print("Error: Image not found or unable to read image")
        return None
    resized_image = cv2.resize(image_to_be_predicted, (28, 28))
    scaled_img = resized_image / 255.0

    # Predict the label
    input_prediction = model.predict(scaled_img.reshape(1, 28, 28))
    predicted_label = np.argmax(input_prediction)

    return predicted_label

# Test the prediction function
image_path = input("Enter the path of the image to be predicted: ")
predicted_label = predict_image(image_path)
if predicted_label is not None:
    print(f"Predicted Label: {predicted_label}")
