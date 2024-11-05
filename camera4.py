import cv2
import numpy as np

# Paths for training images and labels
training_data = [('frame.jpg', 0), ('carlos.jpg', 1)]  # Replace with paths and labels for each person

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare the images and labels for training
faces = []
labels = []
for image_path, label in training_data:
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    # Check that exactly one face is detected, then add it to training data
    if len(detected_faces) == 1:
        (x, y, w, h) = detected_faces[0]
        faces.append(gray_image[y:y+h, x:x+w])
        labels.append(label)
    else:
        print(f"Error in {image_path}: No face or multiple faces detected.")

# Train the recognizer with multiple faces
if faces and labels:
    recognizer.train(faces, np.array(labels))
    print("Training completed with multiple faces.")
else:
    print("Error: No valid faces for training.")
    exit()

# Start the real-time video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection and recognition
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the current frame
    detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    # Process each detected face
    for (x, y, w, h) in detected_faces:
        face_region = gray_frame[y:y+h, x:x+w]
        
        # Predict the label of the face
        label, confidence = recognizer.predict(face_region)
        
        # Check if the detected face is recognized or "Unknown"
        if confidence < 70:  # Adjust confidence threshold as needed
            if label == 0:
                label_text = "Om"
            elif label == 1:
                label_text = "Carlos"
            else:
                label_text = "Unknown"
            color = (0, 255, 0)  # Green for recognized face
        else:
            label_text = "Unknown"
            color = (0, 0, 255)  # Red for unrecognized face
        
        # Draw rectangle around the face and put the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the frame with detection and recognition
    cv2.imshow('Face Recognition', frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
