import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import os
import objc
from Foundation import NSURL
from AppKit import NSOpenPanel, NSApplication, NSApp

# Function to open macOS file dialog using PyObjC
def mac_open_file_dialog(allow_multiple=False):
    panel = NSOpenPanel.openPanel()
    panel.setAllowsMultipleSelection_(allow_multiple)
    panel.setCanChooseFiles_(True)
    panel.setCanChooseDirectories_(False)

    # Run the dialog and capture response
    if panel.runModal() == 1:
        # Return file paths as a list of strings
        return [str(url.path()) for url in panel.URLs()]
    else:
        return []

# Main function to perform face match comparison
def face_match_comparison():
    # Function to load and compare images
    def compare_images():
        # Instructions message
        messagebox.showinfo("Instructions", "Please select reference images first. Then you will be prompted to select the unknown image.")

        # Select reference images
        file_paths1 = mac_open_file_dialog(allow_multiple=True)
        if not file_paths1:
            messagebox.showerror("Error", "No reference images selected!")
            return

        reference_encodings = []
        reference_files = []  # Store the reference image filenames
        for file_path in file_paths1:
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"Invalid path: {file_path}")
                return
            img = cv2.imread(file_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_encodings = face_recognition.face_encodings(rgb_img)
            if img_encodings:
                reference_encodings.append(img_encodings[0])
                reference_files.append(os.path.basename(file_path))  # Store the filename
            else:
                messagebox.showwarning("Warning", f"No face detected in reference image: {file_path}")

        if not reference_encodings:
            messagebox.showerror("Error", "No faces detected in any reference images!")
            return

        # Instructions for selecting unknown image
        messagebox.showinfo("Instructions", "Now, please select the unknown image to compare.")

        # Select unknown image
        file_paths2 = mac_open_file_dialog(allow_multiple=False)
        if not file_paths2:
            messagebox.showerror("Error", "No image selected for the unknown image!")
            return
        file_path2 = file_paths2[0]

        img2 = cv2.imread(file_path2)
        rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img_encoding2 = face_recognition.face_encodings(rgb_img2)

        if not img_encoding2:
            messagebox.showerror("Error", "No face detected in the unknown image!")
            return

        # Compare unknown image with each reference encoding
        match_found = False
        matched_image = ""
        for ref_encoding, ref_file in zip(reference_encodings, reference_files):
            result = face_recognition.compare_faces([ref_encoding], img_encoding2[0])
            if result[0]:
                match_found = True
                matched_image = ref_file  # Store the matched reference image filename
                break

        # Display the result
        if match_found:
            messagebox.showinfo("Result", f"The face matches the reference image: {matched_image}")
        else:
            messagebox.showinfo("Result", "The face does not match any of the reference images.")

    # Set up Tkinter window
    window = tk.Tk()
    window.title("Face Recognition Image Comparison")
    window.geometry("600x400")  # Set the window size to be larger
    window.configure(bg="#262434")  # Set a light background color for better contrast

    # Instructions label
    instruction_label = tk.Label(window, text="Click the button to select reference images and compare with an unknown image.",
                                 font=("Helvetica", 14), bg="#262434")
    instruction_label.pack(pady=20)

    # Button to trigger the comparison
    compare_button = tk.Button(window, text="Select Images and Compare", command=compare_images, 
                               font=("Helvetica", 16), bg="#262434", fg="black", height=2, width=20)
    compare_button.pack(pady=20)

    window.mainloop()






