import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf 
from tensorflow.keras.models import load_model
from cvzone.ClassificationModule import Classifier

# Load and compile the model
model_path = "Model/keras_model.h5"
model = load_model(model_path, compile=False)  # Load without compilation first
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compile manually

# Initialize the classifier with the compiled model
classifier = Classifier(model_path, "Model/labels.txt")

# Open webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Define class labels
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z', 'Mariem', 'Stop', 'OK', 'Hello',
    '1', '2', '3', '4', '5', '7', '8', '9', '_'
]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.size == 0:
            print("Invalid hand crop detected, skipping frame.")
            continue

        aspectRatio = h / w
        try:
            if aspectRatio > 1:
                # Adjust image size while keeping aspect ratio
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Predict using the classifier
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(f"Prediction: {labels[index]}")

            # Draw results
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            # Show images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print(f"Error processing frame: {e}")

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
