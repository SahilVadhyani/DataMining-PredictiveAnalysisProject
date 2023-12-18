# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt


# Seting random seed for reproducibility
tf.random.set_seed(42)

# Defining paths
train_path = "/content/drive/MyDrive/Pneumonia dataset/train"
test_path = "/content/drive/MyDrive/Pneumonia dataset/test"

# Image data augmentation 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Loading and preprocessing the training data
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(64, 64),  # adjust the target size based on your requirements
    batch_size=32,
    class_mode='binary',
    classes=['NORMAL', 'PNEUMONIA']
)

# Loading and preprocessing the test data
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    classes=['NORMAL', 'PNEUMONIA']
)

# Building a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))


# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_generator, epochs=4, validation_data=test_generator)

# Evaluating the model
evaluation = model.evaluate(test_generator)
print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

history = model.fit(train_generator, epochs=4, validation_data=test_generator)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Saving the model
model.save("pneumonia_detection_model.h5")
