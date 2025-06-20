import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import time
import warnings
import datetime
import pyttsx3
import os
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load ML model
try:
    model = keras.models.load_model("model.h5")
except FileNotFoundError:
    print("Error: model.h5 not found. Please provide an ISL-trained model.")
    exit(1)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Alphabet mapping
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase) + [' ', '.']
labels_dict = {i: c for i, c in enumerate(alphabet)}

# Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# Initialize buffers and history
stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""
last_registered_time = time.time()
registration_delay = 2.5  # Seconds between character registrations
expected_features = 42

# File for storing sentences
log_file = "isl_sentences.txt"

def save_sentence(sentence):
    if sentence:
        with open(log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {sentence}\n")

# GUI Setup
root = tk.Tk()
root.title("Indian Sign Language Translator")
root.geometry("1400x800")
root.configure(bg="white")
root.resizable(False, False)

# Custom Colors and Fonts
PRIMARY_COLOR = "#2ecc71"    # Green
SECONDARY_COLOR = "#3498db"  # Blue
ACCENT_COLOR = "#e74c3c"     # Red
BG_COLOR = "white"
CARD_COLOR = "#f8f9fa"
TEXT_COLOR = "#2c3e50"
BORDER_COLOR = "#dee2e6"

TITLE_FONT = ("Helvetica", 32, "bold")
HEADING_FONT = ("Arial", 24, "bold")
BODY_FONT = ("Arial", 18)
BUTTON_FONT = ("Arial", 16, "bold")

# Header Section
header_frame = Frame(root, bg=BG_COLOR)
header_frame.pack(pady=20)

# Load logo (optional)
try:
    logo_img = Image.open("logo.png").resize((80, 80))
    logo_photo = ImageTk.PhotoImage(logo_img)
    logo_label = Label(header_frame, image=logo_photo, bg=BG_COLOR)
    logo_label.grid(row=0, column=0, padx=10)
except:
    pass  # Continue without logo

title_label = Label(header_frame,
                    text="Indian Sign Language Translator",
                    font=TITLE_FONT,
                    fg=TEXT_COLOR,
                    bg=BG_COLOR)
title_label.grid(row=0, column=1, padx=10)

# Main Content Layout
main_frame = Frame(root, bg=BG_COLOR)
main_frame.pack(pady=20)

# Video Feed Section
video_container = Frame(main_frame,
                        bg=BG_COLOR,
                        bd=4,
                        relief="ridge",
                        width=640,
                        height=480,
                        highlightbackground=BORDER_COLOR,
                        highlightthickness=2)
video_container.grid(row=0, column=0, padx=40, pady=10)
video_container.grid_propagate(False)

video_label = Label(video_container, bg=BG_COLOR)
video_label.pack(expand=True, fill="both")

# Results Panel
results_frame = Frame(main_frame, bg=BG_COLOR)
results_frame.grid(row=0, column=1, padx=40, pady=10)

# Detection Card
detection_card = Frame(results_frame,
                      bg=CARD_COLOR,
                      width=400,
                      height=200,
                      highlightbackground=BORDER_COLOR,
                      highlightthickness=1)
detection_card.pack(pady=20, fill="x")
Label(detection_card,
      text="CURRENT DETECTION",
      font=HEADING_FONT,
      bg=CARD_COLOR,
      fg=TEXT_COLOR).pack(pady=10)
current_alphabet = StringVar(value="---")
Label(detection_card,
      textvariable=current_alphabet,
      font=("Arial", 48, "bold"),
      bg=CARD_COLOR,
      fg=PRIMARY_COLOR).pack(pady=10)

# Word Card
word_card = Frame(results_frame,
                 bg=CARD_COLOR,
                 width=400,
                 height=150,
                 highlightbackground=BORDER_COLOR,
                 highlightthickness=1)
word_card.pack(pady=20, fill="x")
Label(word_card,
      text="CURRENT WORD",
      font=HEADING_FONT,
      bg=CARD_COLOR,
      fg=TEXT_COLOR).pack(pady=5)
current_word = StringVar(value="---")
Label(word_card,
      textvariable=current_word,
      font=BODY_FONT,
      bg=CARD_COLOR,
      fg=SECONDARY_COLOR).pack(pady=5)

# Sentence Card
sentence_card = Frame(results_frame,
                     bg=CARD_COLOR,
                     width=400,
                     height=200,
                     highlightbackground=BORDER_COLOR,
                     highlightthickness=1)
sentence_card.pack(pady=20, fill="x")
Label(sentence_card,
      text="TRANSLATED SENTENCE",
      font=HEADING_FONT,
      bg=CARD_COLOR,
      fg=TEXT_COLOR).pack(pady=5)
current_sentence = StringVar(value="---")
Label(sentence_card,
      textvariable=current_sentence,
      font=BODY_FONT,
      bg=CARD_COLOR,
      fg=TEXT_COLOR,
      wraplength=380,
      justify="left").pack(pady=5)

# Control Buttons
button_frame = Frame(root, bg=BG_COLOR)
button_frame.pack(pady=30)

def reset_sentence():
    global word_buffer, sentence
    if sentence.strip() or word_buffer.strip():
        save_sentence(sentence.strip() + (" " + word_buffer.strip() if word_buffer.strip() else ""))
    word_buffer = ""
    sentence = ""
    current_word.set("---")
    current_sentence.set("---")
    current_alphabet.set("---")

def toggle_pause():
    if pause_button.cget('text') == "Pause":
        pause_button.config(text="Resume")
    else:
        pause_button.config(text="Pause")

def speak_text(text):
    if text and text != "---":
        def tts_thread():
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=tts_thread, daemon=True).start()

# Buttons
Button(button_frame,
       text="Reset Session",
       font=BUTTON_FONT,
       command=reset_sentence,
       bg=ACCENT_COLOR,
       fg="white",
       padx=20,
       pady=10,
       bd=0,
       activebackground="#c0392b").grid(row=0, column=0, padx=15)

pause_button = Button(button_frame,
                     text="Pause",
                     font=BUTTON_FONT,
                     command=toggle_pause,
                     bg=SECONDARY_COLOR,
                     fg="white",
                     padx=20,
                     pady=10,
                     bd=0,
                     activebackground="#2980b9")
pause_button.grid(row=0, column=1, padx=15)

Button(button_frame,
       text="Speak Sentence",
       font=BUTTON_FONT,
       command=lambda: speak_text(current_sentence.get()),
       bg=PRIMARY_COLOR,
       fg="white",
       padx=20,
       pady=10,
       bd=0,
       activebackground="#27ae60").grid(row=0, column=2, padx=15)

# Hover Effects
def on_enter(e):
    e.widget['bg'] = e.widget.cget('activebackground')
def on_leave(e):
    original_color = ACCENT_COLOR if e.widget.cget('text') == "Reset Session" else \
                   SECONDARY_COLOR if e.widget.cget('text') == "Pause" else PRIMARY_COLOR
    e.widget['bg'] = original_color

for btn in button_frame.winfo_children():
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

# Handle Enter key to append word to sentence
last_enter_time = 0
enter_cooldown = 0.5  # Seconds

def handle_enter(event):
    global word_buffer, sentence, last_enter_time
    current_time = time.time()
    if current_time - last_enter_time < enter_cooldown:
        return
    last_enter_time = current_time
    if word_buffer.strip():
        sentence += word_buffer + " "
        speak_text(word_buffer)
        word_buffer = ""
        current_word.set("---")
        current_sentence.set(sentence.strip() or "---")
        sentence_card.config(bg="#d4edda")
        root.after(200, lambda: sentence_card.config(bg=CARD_COLOR))

# Bind Enter key to the root window
root.bind('<Return>', handle_enter)

# Video Processing
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    def process_frame():
        global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            root.after(10, process_frame)
            return

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)

        if pause_button.cget('text') == "Resume":
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = img_tk
            video_label.configure(image=img_tk)
            root.after(10, process_frame)
            return

        # To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        debug_image = copy.deepcopy(image)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                df = pd.DataFrame(pre_processed_landmark_list).transpose()

                # Predict the sign language
                predictions = model.predict(df, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)
                try:
                    predicted_character = labels_dict[predicted_classes[0]]
                except KeyError:
                    predicted_character = ' '  # Fallback for invalid predictions

                # Stabilization
                stabilization_buffer.append(predicted_character)
                if len(stabilization_buffer) > 30:
                    stabilization_buffer.pop(0)

                if stabilization_buffer.count(predicted_character) > 25:
                    current_time = time.time()
                    if current_time - last_registered_time > registration_delay:
                        stable_char = predicted_character
                        last_registered_time = current_time
                        current_alphabet.set(stable_char)

                        if stable_char == ' ':
                            if word_buffer.strip():
                                sentence += word_buffer + " "
                                speak_text(word_buffer)  # Speak the completed word
                            word_buffer = ""
                            current_word.set("---")
                            current_sentence.set(sentence.strip() or "---")
                        elif stable_char == '.':
                            if word_buffer.strip():
                                sentence += word_buffer + ". "
                                speak_text(word_buffer)  # Speak the completed word
                                current_word.set("---")
                                word_buffer = ""
                                final_sentence = sentence.strip()
                                current_sentence.set(final_sentence or "---")
                                if final_sentence:
                                    save_sentence(final_sentence)
                            else:
                                current_alphabet.set("---")
                        else:
                            word_buffer += stable_char
                            current_word.set(word_buffer)
                            current_sentence.set((sentence + word_buffer).strip() or "---")
                            speak_text(stable_char)  # Speak the individual character

        # Display current sign on video
        cv2.putText(image, f"Current Sign: {current_alphabet.get()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 120, 200), 2)

        # Update video feed
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)
        root.after(10, process_frame)

    # Start Application
    try:
        root.focus_set()  # Ensure root window captures key events
        process_frame()
        root.mainloop()
    finally:
        cap.release()
        cv2.destroyAllWindows()