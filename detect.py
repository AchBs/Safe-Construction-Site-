import cv2

# Load the HOG+SVM detector for pedestrians
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the video stream
cap = cv2.VideoCapture('image/video.mp4')

# Loop over the frames of the video
while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()

    # Stop the loop if we have reached the end of the video
    if not ret:
        break

    # Detect pedestrians in the current frame using HOG
    pedestrians, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)

    # Draw rectangles around the detected pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Pedestrian Detection', frame)

    # Wait for a key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
