import cv2

# Initialize global variables
region_selected = False
x1, y1, x2, y2 = 0, 0, 0, 0

# Mouse callback function to select region with right-click
def select(event, x, y, flags, param):
    global x1, y1, x2, y2, region_selected
    
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        region_selected = False
    elif event == cv2.EVENT_MOUSEMOVE and not region_selected:
        x2, y2 = x, y
    elif event == cv2.EVENT_RBUTTONDOWN:
        x2, y2 = x, y
        region_selected = True
        print("Region selected:", x1, y1, x2, y2)

# Main function for region-based noise detection
def rect_noise():
    global x1, y1, x2, y2, region_selected
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("select_region")
    cv2.setMouseCallback("select_region", select)
    
    def reset_selection():
        # Reset selection flags and coordinates
        global region_selected, x1, y1, x2, y2
        region_selected = False
        x1, y1, x2, y2 = 0, 0, 0, 0

    def select_region():
        global x1, y1, x2, y2, region_selected  # Ensure these variables are global
        reset_selection()
        while True:
            _, frame = cap.read()
            if not region_selected and x1 != x2 and y1 != y2:
                # Draw a rectangle as user is dragging to select the region
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("select_region", frame)
            if region_selected:
                # Ensure x1 < x2 and y1 < y2 for correct region selection
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                print("Region confirmed:", x1, y1, x2, y2)
                break
            if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
                cap.release()
                cv2.destroyAllWindows()
                return False  # Exit the entire function

        # Close selection window
        cv2.destroyWindow("select_region")
        return True  # Selection successful

    # Initial region selection
    if not select_region():
        return

    previous_frame = None
    while True:
        _, frame = cap.read()
        current_frame = frame[y1:y2, x1:x2]  # Only process the selected region

        # Convert to grayscale and blur
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if previous_frame is None:
            previous_frame = gray
            continue

        # Compute absolute difference between current frame and previous frame
        frame_diff = cv2.absdiff(previous_frame, gray)
        
        # Threshold to highlight differences
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If motion is detected, draw rectangle and display "MOTION"
        if contours:
            max_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_cnt) > 500:  # Threshold for significant motion
                x, y, w, h = cv2.boundingRect(max_cnt)
                cv2.rectangle(frame, (x + x1, y + y1), (x + w + x1, y + h + y1), (0, 255, 0), 2)
                cv2.putText(frame, "MOTION", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO MOTION", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "NO MOTION", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        
        # Draw the selected region on the main frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        # Display the frame
        cv2.imshow("Motion Detection - Press 'r' to reselect / 'Esc' to exit", frame)
        
        # Update previous frame and check for key events
        previous_frame = gray
        key = cv2.waitKey(1)
        if key == 27:  # Exit on 'Esc'
            break
        elif key == ord('r'):  # Re-select region on 'r' key press
            if not select_region():
                return  # Exit if Esc is pressed during region re-selection
            previous_frame = None  # Reset the previous frame

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

