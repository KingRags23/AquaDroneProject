import cv2
from vision import Vision
import time

vision = Vision()
# Initialize the camera
cap = cv2.VideoCapture(0)
fps = 1000
delay_time = int(1000/fps)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    time.sleep(4)
    if ret:
        resize = frame
        #resize = frame[80:280, 150:330]
        cv2.imwrite("frames/frame.jpg", resize)
        vision.process_image("frames/frame.jpg")
        # Apply image processing
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Display the resulting frame
        cv2.imshow('frame', resize)
        cv2.imshow('edges', edges)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(delay_time) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
