import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mediapipe as mp
import matplotlib.pyplot as plt

# Set parameters
img_height, img_width = 224, 224
batch_size = 32
num_epochs = 10

# Provide the paths to your datasets
train_data_dir = 'img'  
validation_data_dir = 'img'  
model_save_path = 'Trained_data.h5'

# Debug: Print directory contents
print("Training data directory contents:", os.listdir(train_data_dir))
print("Validation data directory contents:", os.listdir(validation_data_dir))

# MediaPipe Hand Detection Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def preprocess_and_detect_hand(image):
    """Preprocess the image and detect hand using MediaPipe"""
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = image.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            x_min = max(x_min - 20, 0)
            y_min = max(y_min - 20, 0)
            x_max = min(x_max + 20, w)
            y_max = min(y_max + 20, h)

            roi = image[y_min:y_max, x_min:x_max]
            return cv2.resize(roi, (img_height, img_width))

    return None

def custom_data_generator(directory):
    """Custom data generator that detects hands in images"""
    datagen = ImageDataGenerator(rescale=1./255)

    for batch in datagen.flow_from_directory(directory, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'):
        images, labels = batch
        processed_images = []
        for img in images:
            detected_hand = preprocess_and_detect_hand(img)
            if detected_hand is not None:
                processed_images.append(detected_hand)
            else:
                # If no hand detected, use the original image (optional)
                processed_images.append(cv2.resize(img, (img_height, img_width)))
        
        yield np.array(processed_images), labels

def get_num_samples(directory):
    """Count the number of samples in the dataset directory"""
    return sum([len(files) for _, _, files in os.walk(directory)])

# Create custom data generators
train_generator = custom_data_generator(train_data_dir)
validation_generator = custom_data_generator(validation_data_dir)

# Get the number of samples
num_train_samples = get_num_samples(train_data_dir)
num_validation_samples = get_num_samples(validation_data_dir)

# Calculate steps per epoch
steps_per_epoch = num_train_samples // batch_size
validation_steps = num_validation_samples // batch_size

# Build the model using MobileNetV2 as the base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.__next__()[1].shape[1], activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Unfreeze some layers of the base model for fine-tuning
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

# Recompile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_fine = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Save the trained model
model.save(model_save_path)

# Plot training history
def plot_history(history, history_fine):
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_history(history, history_fine)
