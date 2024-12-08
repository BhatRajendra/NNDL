import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape((60000, 28 * 28)).astype('float32') / 255
x_test = x_test.reshape((10000, 28 * 28)).astype('float32') / 255

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)


# Display the image
for i in range(10):
    # Pick a random image from the test set
    random_index = np.random.randint(0, x_test.shape[0])
    random_image = x_test[random_index]
    random_label = np.argmax(y_test[random_index]) # True label

    # Reshape for prediction
    random_image = random_image.reshape(1, 28 * 28)

    # Make prediction
    predictions = model.predict(random_image)
    predicted_class = np.argmax(predictions)

    plt.imshow(random_image.reshape(28, 28), cmap='gray')
    plt.title(f'True label: {random_label}, Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()
