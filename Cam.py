import cv2
import numpy as np
from keras.models import load_model

# Constants
IMAGE_SIZE = (128, 128)
GUAVA_TYPES = ['Bangkok Red', 'China Surahi', 'Moti Surahi', 'Choti Surahi', 'Golden Gola', 'China Gola', 'Multani Sada Gola', 'Sadda bahar Gola', 'Larkana Surahi', 
               'Black Guava', 'Hyderabadi Safeeda', 'Strawberry Pink Gola', 'Others'
]

# Load the trained model
model = load_model('GL_type_detection_model.h5')

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame
    resized_frame = cv2.resize(frame, IMAGE_SIZE)

    # Preprocess the frame for the model
    input_frame = np.expand_dims(resized_frame / 255.0, axis=0)

    # Predict mango type
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions)

    # Get contours to identify the leaf
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and draw a circle on the contour with the largest area
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)

        # Filter out small contours (adjust the area threshold as needed)
        if cv2.contourArea(max_contour) > 1000:
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

            # Display the predicted mango type on the frame
            cv2.putText(frame, f'Guava Type: {GUAVA_TYPES[predicted_class]}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # No leaf detected, display a message
            cv2.putText(frame, 'No Guava leaf detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Guava Type Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
