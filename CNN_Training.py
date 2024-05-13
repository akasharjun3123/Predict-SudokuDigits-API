from keras.datasets import mnist
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pickle
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
# STEP 1 : LOADING THE DATA  FROM MNIST DATABASE
try:
    with open('mnist_dataset.pkl', 'rb') as file:
        train_images, train_labels, test_images, test_labels = pickle.load(file)

except FileNotFoundError:
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Save the dataset using pickle
    with open('mnist_dataset.pkl', 'wb') as file:
        pickle.dump((train_images, train_labels, test_images, test_labels), file)

    # Use the loaded data
    train_images, train_labels, test_images, test_labels = train_images, train_labels, test_images, test_labels


train_images=train_images/255
test_images=test_images/255


# STEP 2 : STORE THE FREQUENCY OF EACH NUMBER CLASS IN A LIST
numberOfImages = [np.sum(train_labels == i) for i in range(10)]

# STEP 3 : ADDING DEPTH OF 1 FOR CNN

train_images=train_images.reshape(train_images.shape[0],train_images.shape[1],train_images.shape[2],1)
test_images=test_images.reshape(test_images.shape[0],test_images.shape[1],test_images.shape[2],1)
# print(train_images.shape)
# print(test_images.shape)

dataGenerateed = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10,
                                    )

dataGenerateed.fit(train_images)

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Train the model using the training data generator




# STEP 4: BUILDING THE CNN MODEL
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # 10 output classes for digits 0 through 9
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# STEP 5: TRAINING THE MODEL
model.fit(dataGenerateed.flow(train_images, train_labels), epochs=5, validation_data=(val_images, val_labels))
model.save('Trained_Model.h5')

# STEP 6: EVALUATING THE MODEL
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)

# STEP 7: MAKING PREDICTIONS
predictions = model.predict(test_images)

# Optional: Visualize a few predictions
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"A: {test_labels[i]}, P: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.show()

 
