import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

# Load the trained model
model = load_model(os.path.join('models', 'overloadedVehicles.keras'))

# Function to process a frame and detect objects
def process_frame(frame, cnt):
    folder_name = "violations"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    name = os.path.join(folder_name, "detected_object_" + str(cnt) + ".jpg")
    
    # Resize the frame and normalize it
    resized_frame = tf.image.resize(frame, (256, 256))
    normalized_frame = resized_frame / 255.0

    # Predict using the model
    prediction = model.predict(np.expand_dims(resized_frame / 255, 0))
    if prediction > 0.5:
        # Find contours of the detected object
        contours, _ = cv2.findContours(np.uint8(prediction[0] * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangle around each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save the frame when object is detected
        cv2.imwrite(name, frame)


# Function to process video file
def process_video(video_file):
    cnt = 0
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Couldn't open video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process each frame
        cnt = cnt + 1
        process_frame(frame , cnt)

    cap.release()


# Run the code
if __name__ == "__main__":
    video_file = "input.mp4"  # Change this to the path of your video file
    process_video(video_file)