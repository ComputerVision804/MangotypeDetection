
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# The rest of your TensorFlow code...
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models

from tensorflow import compat

# ...


# Constants
IMAGE_SIZE = (242, 242)
NUM_CLASSES = 4  # Number of grave types: GL-Class_1, GL-Class_2, GL-Class_3, GL-Class_4,.......,GL-Class_13

# Function to load and preprocess images
def load_and_preprocess_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0  # Normalize pixel values to be between 0 and 1
            images.append(img)
            labels.append(label)
    return images, labels

# Load mango type images
Chausa_Desi_images, Chausa_Desi_labels = load_and_preprocess_images('C:\\Users\\Laptop Seller\\Desktop\\Mango\\Database\\Chausa Desi', label=0)
Dasheri_Desi_images, Dasheri_Desi_labels = load_and_preprocess_images('C:\\Users\\Laptop Seller\\Desktop\\Mango\\Database\\Dasheri Desi', label=1)
Langra_Desi_images, Langra_Desi_labels = load_and_preprocess_images('C:\\Users\\Laptop Seller\\Desktop\\Mango\\Database\\Langra Desi', label=2)
Ratol_images, Ratol_labels = load_and_preprocess_images('C:\\Users\\Laptop Seller\\Desktop\\Mango\\Database\\Ratol', label=3)

# Combine different mango type data
all_images = Chausa_Desi_images + Dasheri_Desi_images + Langra_Desi_images + Ratol_images
all_labels = Chausa_Desi_labels + Dasheri_Desi_labels + Langra_Desi_labels + Ratol_labels
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Convert lists to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Build a simple CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=50, validation_data=(X_test, y_test))

# Save the model
model.save('Mango_model.keras')

# Convert the Keras model to TensorFlow Lite format (.tflite)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('GL_type_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)
