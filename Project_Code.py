import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# ------------------ Configuration Variables ------------------ #
imageNumber = 0                          # Index of current slide image
width, height = 1280, 720                # Width and height of the webcam frame
folder_path = "Presentation"             # Folder containing the presentation images
hs, ws = int(120 * 1.2), int(213 * 1.2)  # Size of webcam thumbnail on slide
gestureThreshold = 300                   # Y-coordinate threshold for gesture recognition
buttonPress = False                      # Flag to prevent multiple triggers
buttCounter = 0                          # Counter for button press delay
buttondelay = 10                         # Delay duration for gesture switching

# Annotations (for drawing on slides)
annotations = [[]]                       # List of all annotation strokes
annotationNumber = 0                     # Index of the current annotation stroke
annotationStart = False                 # Flag indicating whether drawing has started

# ------------------ Load Image Paths ------------------ #
valid_exts = ['.png', '.jpg', '.jpeg']   # Valid image extensions
pathImages = sorted([f for f in os.listdir(folder_path)
                     if os.path.splitext(f)[1].lower() in valid_exts])
print(pathImages)

# ------------------ Initialize Webcam ------------------ #
cap = cv2.VideoCapture(0)
cap.set(3, width)  # Set width
cap.set(4, height) # Set height

# ------------------ Hand Detector ------------------ #
detector = HandDetector(detectionCon=0.8, maxHands=1)  # Detect one hand with 80% confidence

# ------------------ Main Loop ------------------ #
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror image for natural interaction

    # Load the current presentation slide
    pathFullimage = os.path.join(folder_path, pathImages[imageNumber])
    imgCurrent = cv2.imread(pathFullimage)

    # Detect hands in the webcam frame
    hands, img = detector.findHands(img)
    
    # Draw a line indicating gesture control threshold
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), thickness=10)

    if hands and buttonPress is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)     # List representing which fingers are up
        cx, cy = hand['center']                # Center of the hand
        lmList = hand['lmList']                # List of 21 hand landmarks

        # Normalize index finger coordinates to slide space
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFingers = xVal, yVal

        # --------------- Slide Navigation (if hand is raised) --------------- #
        if cy <= gestureThreshold:
            # Left swipe gesture
            if fingers == [1, 0, 0, 0, 0]:
                annotationStart = False
                print("left")
                if imageNumber > 0:
                    annotations = [[]]
                    annotationNumber = 0
                    buttonPress = True
                    imageNumber -= 1

            # Right swipe gesture
            elif fingers == [0, 0, 0, 0, 1]:
                annotationStart = False
                print("right")
                if imageNumber < len(pathImages) - 1:
                    annotations = [[]]
                    annotationNumber = 0
                    annotationStart = False
                    buttonPress = True
                    imageNumber += 1

        # --------------- Drawing Pointer --------------- #
        if fingers == [0, 1, 1, 0, 0]:
            # Only index and middle fingers up – show pointer
            cv2.circle(imgCurrent, indexFingers, 12, (0, 0, 255), cv2.FILLED)

        # --------------- Drawing Annotation --------------- #
        if fingers == [0, 1, 0, 0, 0]:
            # Only index finger up – start or continue drawing
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFingers, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFingers)
        else:
            annotationStart = False

        # --------------- Undo Last Annotation --------------- #
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPress = True

    # --------------- Delay to Prevent Multiple Gestures --------------- #
    if buttonPress is True:
        buttCounter += 1
        if buttCounter >= buttondelay:
            buttCounter = 0
            buttonPress = False

    # --------------- Draw All Annotations --------------- #
    for i in range(len(annotations)):
        for j in range(1, len(annotations[i])):
            cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 255), 5)

    # --------------- Overlay Webcam Feed on Slide --------------- #
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:width] = imgSmall  # Place small webcam frame on top right

    # Show both windows
    cv2.imshow('frame', img)         # Webcam frame
    cv2.imshow('Slides', imgCurrent) # Current slide with annotations

    # Quit the app when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
