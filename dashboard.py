import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import time
import random

# Emotion labels
emotions = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("❌ Haarcascade not loaded")
    st.stop()

st.title("😊 Emotion Detection Dashboard (Demo Version)")

# Session state
if "run" not in st.session_state:
    st.session_state.run = False

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Camera"):
        st.session_state.run = True

with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.run = False

frame_window = st.image([])
data = []

# Camera loop
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30)
        )

        current_emotion = None

        for (x, y, w, h) in faces:
            # Demo prediction (random)
            label = random.choice(emotions)
            confidence = round(random.uniform(0.6, 0.95), 2)

            current_emotion = label

            # Draw box + label
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,f"{label} ({confidence})",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        frame_window.image(frame, channels="BGR")

        # Store data
        if current_emotion:
            data.append({
                "time": datetime.now(),
                "emotion": current_emotion
            })

        time.sleep(0.03)

    cap.release()

# Analytics
if data:
    df = pd.DataFrame(data)

    st.subheader("📊 Emotion Data")
    st.write(df.tail())

    st.subheader("📈 Emotion Distribution")
    st.bar_chart(df["emotion"].value_counts())

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="emotions_log.csv",
        mime="text/csv"
    )
