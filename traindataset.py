import cv2
import os
import numpy as np

# Update the path to your image directory
image_directory = r'D:\Masters\SEM 2\SAI2\CODE SAI\dataset\images'  # Use the correct absolute path

# Prepare data
face_images = []
labels = []

# Iterate through images and collect face data
for image_name in os.listdir(image_directory):
    if image_name.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(image_directory, image_name)
        image = cv2.imread(img_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Add to data lists
        face_images.append(gray)

        # Assuming your label is the same for all captured images
        labels.append(0)  # You can set a specific label if needed

# Convert to numpy arrays
X_train = np.array(face_images)
y_train = np.array(labels)

# Create a Local Binary Patterns Histogram face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer
recognizer.train(X_train, y_train)

# Save the model to a file
recognizer.save("face_model.xml")

print("Model trained and saved as face_model.xml.")
