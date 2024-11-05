import cv2
import os

# Create a directory to save your face images
dataset_dir = 'dataset/'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Start video capture
video_capture = cv2.VideoCapture(0)

# Initialize count
count = 0
max_images = 50  # Set the number of images to capture

while count < max_images:
    # Capture a single frame
    ret, frame = video_capture.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('Video', gray)

    # Save the captured image
    image_name = os.path.join(dataset_dir, f'face_{count}.jpg')
    cv2.imwrite(image_name, gray)

    count += 1
    print(f'Saved: {image_name}')

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
print("Image capture complete.")
