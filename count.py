import cv2

# Global variables for the selected region
x1, y1, x2, y2 = -1, -1, -1, -1
drawing = False

# Mouse callback function to select region
def select_region(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing the rectangle
        drawing = True
        x1, y1 = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the rectangle as the mouse moves
            x2, y2 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing the rectangle
        drawing = False
        x2, y2 = x, y

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set up window and mouse callback
cv2.namedWindow("Object Detection")
cv2.setMouseCallback("Object Detection", select_region)

# Create a background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor to get the foreground mask
    fg_mask = background_subtractor.apply(frame)
    
    # Thresholding to create a binary image
    _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to remove noise and fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # If a region is selected, count objects inside it
    if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
        # Ensure the selected region is valid (non-zero width and height)
        if x2 > x1 and y2 > y1:
            # Draw the selected region on the frame (static size)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Crop the frame to the selected region
            region = frame[y1:y2, x1:x2]

            # Find contours (connected components) inside the selected region
            contours, _ = cv2.findContours(thresh[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            object_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Adjust area threshold based on object size
                    object_count += 1
                    # Draw bounding boxes around detected objects inside the region
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display object count inside the selected region
            cv2.putText(region, f'Object Count: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the region with bounding boxes
            cv2.imshow("Region with Objects", region)
        else:
            # If region is invalid (width or height is zero), don't process
            cv2.putText(frame, "Invalid Region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the full frame with the rectangle overlay (if selected)
    cv2.imshow("Object Detection", frame)

    # Exit on pressing the 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
