#p4
#Program 4: Implement a Convolutional Neural Network (CNN) using Python and a deep learning library (PyTorch) 
#to classify images from the CIFAR-10 dataset. Use CIFAR-100 dataset and retrain the model. 

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
# Normalize images to values between 0 and 1
x_train_10, x_test_10 = x_train_10 / 255.0, x_test_10 / 255.0
# One-hot encode labels
y_train_10 = to_categorical(y_train_10, 10)
y_test_10 = to_categorical(y_test_10, 10)

# Load CIFAR-100 dataset for retraining
(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()
# Normalize images to values between 0 and 1
x_train_100, x_test_100 = x_train_100 / 255.0, x_test_100 / 255.0
# One-hot encode labels
y_train_100 = to_categorical(y_train_100, 100)
y_test_100 = to_categorical(y_test_100, 100)

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # First convolutional block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Define input shape and number of classes for CIFAR-10
input_shape = x_train_10.shape[1:]  # (32, 32, 3)
num_classes_10 = 10

# Build and train the model for CIFAR-10
model_10 = build_cnn_model(input_shape, num_classes_10)
model_10.summary()

# Train the model on CIFAR-10 dataset
history_10 = model_10.fit(x_train_10, y_train_10, epochs=20, batch_size=64, validation_data=(x_test_10, y_test_10))

# Evaluate the model on CIFAR-10 test data
test_loss_10, test_acc_10 = model_10.evaluate(x_test_10, y_test_10, verbose=2)
print(f"CIFAR-10 Test accuracy: {test_acc_10:.4f}")

# Define number of classes for CIFAR-100
num_classes_100 = 100

# Rebuild and retrain the model for CIFAR-100
model_100 = build_cnn_model(input_shape, num_classes_100)
model_100.summary()

# Train the model on CIFAR-100 dataset
history_100 = model_100.fit(x_train_100, y_train_100, epochs=20, batch_size=64, validation_data=(x_test_100, y_test_100))

# Evaluate the model on CIFAR-100 test data
test_loss_100, test_acc_100 = model_100.evaluate(x_test_100, y_test_100, verbose=2)
print(f"CIFAR-100 Test accuracy: {test_acc_100:.4f}")

# Plot training and validation accuracy for CIFAR-10
plt.plot(history_10.history['accuracy'], label='CIFAR-10 Training Accuracy')
plt.plot(history_10.history['val_accuracy'], label='CIFAR-10 Validation Accuracy')
plt.title('CIFAR-10 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation accuracy for CIFAR-100
plt.plot(history_100.history['accuracy'], label='CIFAR-100 Training Accuracy')
plt.plot(history_100.history['val_accuracy'], label='CIFAR-100 Validation Accuracy')
plt.title('CIFAR-100 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
