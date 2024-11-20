import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the trained model
model_path = 'Trained_data.h5'  # Replace with your model path
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    exit()

# Define the labels (classes) for the model predictions
labels = ['Hello', 'el2', 'el3', 'Sorry', 'Why']  # Replace with your actual class names

# Define the input image size
img_height, img_width = 224, 224

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Capture input from a video file
video_path = 'WIN_20240830_10_24_13_Pro.mp4'  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("End of video reached or error reading frame.")
        break

    # Flip the frame horizontally for a mirror-like effect (optional)
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as MediaPipe expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box of the hand
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            # Expand the bounding box slightly for better coverage
            x_min = max(x_min - 20, 0)
            y_min = max(y_min - 20, 0)
            x_max = min(x_max + 20, frame.shape[1])
            y_max = min(y_max + 20, frame.shape[0])

            # Extract the region of interest (ROI) where the hand is located
            roi = frame[y_min:y_max, x_min:x_max]

            # Preprocess the ROI for the model
            img = cv2.resize(roi, (img_height, img_width))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Make a prediction
            predictions = model.predict(img)

            # Extract the predicted probabilities and find the class with the highest probability
            predicted_class_index = np.argmax(predictions[0])
            predicted_label = labels[predicted_class_index]
            predicted_probability = predictions[0][predicted_class_index]

            # Display the prediction and its probability on the frame
            cv2.putText(frame, f"{predicted_label} ({predicted_probability:.2f})", 
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                        (255, 0, 0), 2)

    # Display the frame with the hand landmarks and prediction
    cv2.imshow('Video', frame)

    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close the window
cap.release()
cv2.destroyAllWindows()