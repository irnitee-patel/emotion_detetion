import streamlit as st
import cv2
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from datetime import datetime
from collections import deque
import time

# ==============================
# LOAD MODEL
# ==============================
model = load_model("model/emotion_model.hdf5", compile=False)

emotions = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# Load Haarcascade (use relative path)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    st.error("❌ Haarcascade not loaded")
    st.stop()

# Emotion smoothing buffer
emotion_buffer = deque(maxlen=5)

st.title("😊 Emotion Detection Dashboard")

# ==============================
# SESSION STATE
# ==============================
if "run" not in st.session_state:
    st.session_state.run = False

if "data" not in st.session_state:
    st.session_state.data = []

# ==============================
# BUTTONS
# ==============================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Camera"):
        st.session_state.run = True

with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.run = False

frame_window = st.empty()

# ==============================
# CAMERA LOOP
# ==============================
if st.session_state.run:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # FIX for Windows

    if not cap.isOpened():
        st.error("❌ Cannot access camera")
        st.stop()

    while st.session_state.run:
        ret, frame = cap.read()

        if not ret:
            st.error("❌ Camera not working")
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=6,
            minSize=(50, 50)
        )

        current_emotion = None

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
                continue

            # ==============================
            # FIXED INPUT SHAPE
            # ==============================
            face = cv2.resize(face, (64, 64))
            face = face.astype("float32") / 255.0
            face = np.reshape(face, (1, 64, 64, 1))

            pred = model.predict(face, verbose=0)[0]
            confidence = np.max(pred)

            emotion_buffer.append(np.argmax(pred))
            smoothed = max(set(emotion_buffer), key=emotion_buffer.count)

            label = emotions[smoothed] if confidence > 0.5 else "Uncertain"
            current_emotion = label

            # Draw bounding box
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame,
                        f"{label} ({confidence:.2f})",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)

        # Show frame in Streamlit
        frame_window.image(frame, channels="BGR", use_container_width=True)

        # Store data
        if current_emotion:
            st.session_state.data.append({
                "time": datetime.now(),
                "emotion": current_emotion
            })

        time.sleep(0.03)

    cap.release()

# ==============================
# ANALYTICS
# ==============================
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)

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
