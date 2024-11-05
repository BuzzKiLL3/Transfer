import cv2
import os
import time  # Import time module for delays

# Create a directory to save your face images
dataset_dir = 'images/'
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
    
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)  # Show the color frame

    # Save the captured image
    color_image_name = os.path.join(dataset_dir, f'face_{count}.jpg')
    cv2.imwrite(color_image_name, frame)  # Save color image

    print(f'Saved: {color_image_name}')

    # Wait for a short time (e.g., 1 second) to allow you to tilt your face
    time.sleep(1)  # Delay for 1 second

    count += 1

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
print("Image capture complete.")
