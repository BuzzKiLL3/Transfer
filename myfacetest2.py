import cv2
import pickle

# Load the trained model
with open("faces_data.pkl", "rb") as f:
    recognizer = pickle.load(f)

# Create a dictionary for names
name_dict = {
    0: "Om",  # Adjust the label according to your dataset
    # Add other labels if necessary
}

# Start video capture
video_capture = cv2.VideoCapture(0)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture a single frame
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Loop through each face found in the frame
    for (x, y, w, h) in faces:
        # Get the face ROI (Region of Interest)
        face_roi = gray_frame[y:y+h, x:x+w]

        # Recognize the face
        label, confidence = recognizer.predict(face_roi)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the name corresponding to the label
        name = name_dict.get(label, "Unknown")
        
        # Label the face with a name
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
