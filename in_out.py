import cv2
from datetime import datetime

def in_out():
    # Initialize video capture from the webcam
    cap = cv2.VideoCapture(0)
    
    # Variables to track movement direction
    right, left, up, down = False, False, False, False
    prev_x, prev_y = None, None  # Variables to track the previous position for direction calculation
    
    while True:
        # Read and flip frames from the webcam
        _, frame1 = cap.read()
        frame1 = cv2.flip(frame1, 1)
        
        _, frame2 = cap.read()
        frame2 = cv2.flip(frame2, 1)
        
        # Compute the difference between the two frames
        diff = cv2.absdiff(frame2, frame1)
        diff = cv2.blur(diff, (5, 5))
        
        # Convert the difference to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to the difference
        _, threshd = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(threshd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        x, y = 0, 0  # Default values for x, y
        
        if len(contours) > 0:
            # Find the largest contour by area
            max_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_cnt)
            
            # Draw a rectangle around the detected motion
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Check for movement direction
            direction = ""
            
            if prev_x is not None and prev_y is not None:
                # Check horizontal movement
                if x > prev_x + 10:  # Moving right
                    direction = "Moving Right"
                elif x < prev_x - 10:  # Moving left
                    direction = "Moving Left"
                
                # Check vertical movement
                if y > prev_y + 10:  # Moving down
                    direction = "Moving Down"
                elif y < prev_y - 10:  # Moving up
                    direction = "Moving Up"
            
            # Display the direction of movement on the frame
            if direction:
                cv2.putText(frame1, direction, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Update previous x, y values for the next frame
            prev_x, prev_y = x, y

            # If movement is detected, save the image
            cv2.putText(frame1, "MOTION", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        
        # Display the frame with movement direction on the screen
        cv2.imshow("Movement Detection", frame1)
        
        # Exit the loop if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break
    
    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
