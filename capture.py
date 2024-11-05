import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture a single frame
ret, frame = cap.read()

# Save the captured frame
cv2.imwrite('my_face.jpg', frame)

# Release the webcam
cap.release()
cv2.destroyAllWindows()
