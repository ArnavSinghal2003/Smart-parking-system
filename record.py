import cv2
from datetime import datetime
import os

def record():
    # Check and create the directory if it doesn't exist
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    
    # Initialize the camera capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Use 'XVID' codec for compatibility and save the video in .mp4 format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec
    
    # Set resolution (optional but may improve compatibility)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create a unique filename based on the current time
    out = cv2.VideoWriter(f'recordings/{datetime.now().strftime("%H-%M-%S")}.mp4', 
                          fourcc, 20.0, (640, 480))
    
    while True:
        ret, frame = cap.read()
        
        # Check if the frame was captured correctly
        if not ret:
            print("Failed to grab frame.")
            break

        # Overlay timestamp on the frame
        cv2.putText(frame, f'{datetime.now().strftime("%D-%H-%M-%S")}', 
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write the frame to the video file
        out.write(frame)
        
        # Display the frame
        cv2.imshow("Press 'Esc' to stop", frame)
        
        # Exit if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release the video capture and output objects, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

