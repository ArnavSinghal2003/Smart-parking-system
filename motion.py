import cv2

def noise():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame1 = cap.read()
        _, frame2 = cap.read()

        # Calculate difference between two frames
        diff = cv2.absdiff(frame2, frame1)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff = cv2.blur(diff, (5, 5))

        # Apply threshold to detect significant changes
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Find contours of the threshold image
        contr, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are detected
        if len(contr) > 0:
            max_cnt = max(contr, key=cv2.contourArea)  # Find the largest contour
            x, y, w, h = cv2.boundingRect(max_cnt)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "MOTION", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        else:
            cv2.putText(frame1, "NO-MOTION", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("esc. to exit", frame1)

        # Exit if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
