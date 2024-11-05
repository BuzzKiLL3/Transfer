import cv2
import face_recognition

# Load your face image and create an encoding
my_image = face_recognition.load_image_file("my_face.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]

# Initialize the video capture (0 for the first camera)
cap = cv2.VideoCapture(0)

# Create an array for known face encodings and their names
known_face_encodings = [my_face_encoding]
known_face_names = ["My Face"]

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB (face_recognition uses RGB instead of BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"  # Default name if no match is found

        # If a match is found, use the known name
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Label the recognized face
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
