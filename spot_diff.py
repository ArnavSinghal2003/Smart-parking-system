import cv2
import time
from skimage.metrics import structural_similarity
from datetime import datetime
import beepy

def spot_diff(frame1, frame2):
    frame1 = frame1[1]
    frame2 = frame2[1]

    # Convert to grayscale
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Blur the images to remove noise
    g1 = cv2.blur(g1, (2, 2))
    g2 = cv2.blur(g2, (2, 2))

    # Compute structural similarity
    (score, diff) = structural_similarity(g1, g2, full=True)
    print("Image similarity:", score)

    # Normalize the difference image
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image
    thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)[1]

    # Find contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [c for c in contours if cv2.contourArea(c) > 50]

    if len(contours):
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        print("Nothing stolen")
        return 0

    # Show the difference and original image
    cv2.imshow("diff", thresh)
    cv2.imshow("win1", frame1)

    # Play a beep sound
    beepy.beep(sound=4)

    # Save the image with detected differences
    cv2.imwrite("stolen/" + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + ".jpg", frame1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
