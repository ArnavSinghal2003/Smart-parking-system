import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageTk
from in_out import in_out
from motion import noise
from rect_noise import rect_noise
from record import record
from face_recog import identify_faces
from face_diff import face_match_comparison
import subprocess
import cv2
import numpy as np

# Fire detection function (same as before)
def fire_detection(video_path="Smoke2.mp4"):
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

# Create the main window
window = tk.Tk()
window.title("Smart CCTV")
window.iconphoto(False, tk.PhotoImage(file='icons/icon.png'))
window.geometry('1080x1000')
window.configure(bg='#262434')  # Dark background for contrast

frame1 = tk.Frame(window, bg='#262434')

# Icon
icon = Image.open('icons/Smart-Survelliance.jpg')
icon = icon.resize((500, 300), Image.Resampling.LANCZOS)
icon = ImageTk.PhotoImage(icon)
label_icon = tk.Label(frame1, image=icon, bg='#262434')
label_icon.grid(row=1, pady=(5, 10), column=1, columnspan=3)  # Centering the icon

# Title label positioned right below the image
label_title = tk.Label(frame1, text="A Project by Arnav Khare, Arnav Singhal & Shardul Shukla", bg='#262434', fg='#EBEFF2')
label_font = font.Font(size=24, weight='bold', family='Poppins')  # Font settings
label_title['font'] = label_font
label_title.grid(row=2, pady=(5, 20), column=1, columnspan=3)  # Centering the title

# Button images and font with adjustments
btn_font = font.Font(size=18, weight='bold')  # Slightly smaller font size

btn1_image = ImageTk.PhotoImage(Image.open('icons/lamp.png').resize((50, 50), Image.Resampling.LANCZOS))
btn2_image = ImageTk.PhotoImage(Image.open('icons/dotted_rectangle.png').resize((50, 50), Image.Resampling.LANCZOS))
btn3_image = ImageTk.PhotoImage(Image.open('icons/motion.jpeg').resize((50, 50), Image.Resampling.LANCZOS))
btn4_image = ImageTk.PhotoImage(Image.open('icons/recording.png').resize((40, 40), Image.Resampling.LANCZOS))
btn5_image = ImageTk.PhotoImage(Image.open('icons/exit.jpg').resize((40, 40), Image.Resampling.LANCZOS))
btn6_image = ImageTk.PhotoImage(Image.open('icons/incognito.jpeg').resize((50, 40), Image.Resampling.LANCZOS))
btn7_image = ImageTk.PhotoImage(Image.open('icons/a.png').resize((70, 50), Image.Resampling.LANCZOS))
btn8_image = ImageTk.PhotoImage(Image.open('icons/camera-icon.png').resize((50, 50), Image.Resampling.LANCZOS))
btn_fire_image = ImageTk.PhotoImage(Image.open('icons/fire.jpg').resize((50, 50), Image.Resampling.LANCZOS))
btn_face_recognition_img = ImageTk.PhotoImage(Image.open('icons/a.jpg').resize((50, 50), Image.Resampling.LANCZOS))

# Define button styles for consistency
btn_style = {
    'height': 90,
    'width': 180,
    'font': btn_font,
    'activebackground': '#525068',
    'relief': 'flat',
    'bd': 2,
    'highlightthickness': 0,
    'compound': 'left',
    'fg': '#262434',
    'bg': '#525068'
}

# Buttons with symmetric grid positions and padding adjustments
btn_face_recognition = tk.Button(frame1, text="Face Match", command=face_match_comparison, image=btn_face_recognition_img, **btn_style) 
btn_face_recognition.grid(row=3, column=1, pady=(20, 20), padx=(10, 10))  

btn2 = tk.Button(frame1, text='Rectangle', command=rect_noise, image=btn2_image, **btn_style)
btn2.grid(row=3, column=2, pady=(20, 20), padx=(10, 10))

btn7 = tk.Button(frame1, text="Identify", command=identify_faces, image=btn7_image, **btn_style)
btn7.grid(row=3, column=3, pady=(20, 20), padx=(10, 10))

btn3 = tk.Button(frame1, text='Noise', command=noise, image=btn3_image, **btn_style)
btn3.grid(row=4, column=1, pady=(20, 20), padx=(10, 10))

btn6 = tk.Button(frame1, text='Direction', command=in_out, image=btn6_image, **btn_style)
btn6.grid(row=4, column=2, pady=(20, 20), padx=(10, 10))

btn4 = tk.Button(frame1, text='  Record', command=record, image=btn4_image, **btn_style)
btn4.grid(row=4, column=3, pady=(20, 20), padx=(10, 10))

# New button to trigger the fire detection
btn_fire = tk.Button(frame1, text="Fire", command=lambda: fire_detection("Smoke2.mp4"), image=btn_fire_image, **btn_style)
btn_fire.grid(row=5, column=2, pady=(20, 20), padx=(10, 10))

# New button to run the count.py script
def run_count_script():
    subprocess.Popen(['python3', 'count.py'])

btn8 = tk.Button(frame1, text="Count", command=run_count_script, image=btn8_image, **btn_style)
btn8.grid(row=5, column=1, pady=(20, 20), padx=(10, 10))

# Exit button with red color for urgency
btn5 = tk.Button(frame1, text='Exit', command=window.quit, image=btn5_image, **btn_style)
btn5.config(bg='#FF4C4C')  # Specific bg color for exit button
btn5.grid(row=5, column=3, pady=(20, 20), padx=(10, 10))

# Pack and display the window
frame1.pack()
window.mainloop()
