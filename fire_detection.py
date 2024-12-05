import cv2
import numpy as np

def detect_fire(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color range for fire (reddish-orange color)
        lower_fire = np.array([0, 50, 200], dtype=np.uint8)
        upper_fire = np.array([25, 255, 255], dtype=np.uint8)
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

        # Find contours for fire detection
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 5000:  # Adjust area threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Fire Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display the frame with fire detection
        cv2.imshow("Fire Detection", frame)

        # Exit if 'Esc' key is pressed
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function with a video file path
detect_fire("Smoke2.mp4")

