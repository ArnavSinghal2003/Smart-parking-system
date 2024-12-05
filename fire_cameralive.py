import cv2
import numpy as np

def detect_fire():
    # Open the system camera (0 for the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color range for fire (reddish-orange color)
        # These values can be adjusted for better sensitivity
        lower_fire = np.array([5, 50, 200], dtype=np.uint8)  # Adjusted lower range for better sensitivity
        upper_fire = np.array([20, 255, 255], dtype=np.uint8)  # Adjusted upper range for better sensitivity
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

        # Apply Gaussian blur to smooth the image and reduce noise
        fire_mask = cv2.GaussianBlur(fire_mask, (15, 15), 0)

        # Find contours for fire detection
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fire_detected = False

        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Increased area threshold to filter out small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Fire Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                fire_detected = True

        # If no fire is detected, you can optionally add a message or action
        if not fire_detected:
            cv2.putText(frame, "No Fire Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the frame with fire detection
        cv2.imshow("Fire Detection", frame)

        # Exit if 'Esc' key is pressed
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function to detect fire live from the system camera
detect_fire()
