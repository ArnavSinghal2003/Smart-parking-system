import cv2
from datetime import datetime

def record():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create a unique filename based on the current time
    out = cv2.VideoWriter(f'recordings/{datetime.now().strftime("%H-%M-%S")}.avi', 
                          fourcc, 20.0, (640, 480))
    
    while True:
        _, frame = cap.read()
        
        # Overlay timestamp on the frame
        cv2.putText(frame, f'{datetime.now().strftime("%D-%H-%M-%S")}', 
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write the frame to the video file
        out.write(frame)
        
        # Display the frame
        cv2.imshow("esc. to stop", frame)
        
        # Exit if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release the video capture and output objects, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
