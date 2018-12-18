# Image Classification

# Import libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
import matplotlib.pyplot as plt

# Initalize CNN
classifier = Sequential()

# Add 2 convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add 2 more convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add 2 more convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add global average pooling layer
classifier.add(GlobalAveragePooling2D())

# Add full connection
classifier.add(Dense(units=2, activation='softmax'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'data/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'data/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

history = classifier.fit_generator(
        train_set,
        samples_per_epoch=8671,
        epochs=10,
        validation_data=test_set,
        validation_steps=300)

# Save the model
classifier.save("model5.h5")
print("Saved model success")


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()