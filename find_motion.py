import cv2
from spot_diff import spot_diff
import time
import numpy as np

def find_motion():
    cap = cv2.VideoCapture(0)
    check = []
    print("waiting for 2 seconds")
    time.sleep(2)
    
    # Read the initial frame
    _, frame1 = cap.read()
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        _, frame2 = cap.read()
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate the difference between the two frames
        diff = cv2.absdiff(frame1_gray, frame2_gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        # Initialize flags
        is_start_done = False
        motion_detected = False
        
        # Filter out small contours
        contours = [c for c in contours if cv2.contourArea(c) > 25]

        if len(contours) > 5:
            # Motion detected
            cv2.putText(thresh, "motion detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            motion_detected = True
            is_start_done = False
        elif motion_detected and len(contours) < 3:
            if not is_start_done:
                start = time.time()
                is_start_done = True

            end = time.time()

            print(end - start)
            if (end - start) > 4:
                _, frame2 = cap.read()
                cap.release()
                cv2.destroyAllWindows()
                x = spot_diff(frame1, frame2)
                if x == 0:
                    print("running again")
                    return
                else:
                    print("found motion")
                    return
        else:
            # No motion detected
            cv2.putText(thresh, "no motion detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        
        # Display the thresholded frame
        cv2.imshow("winname", thresh)
        
        # Update frame1 for the next iteration
        _, frame1 = cap.read()
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # Break the loop on pressing 'Esc'
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
