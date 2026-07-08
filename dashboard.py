import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time

from datetime import datetime
from collections import deque
from tensorflow.keras.models import load_model

# ===========================
# PAGE CONFIGURATION
# ===========================

st.set_page_config(
    page_title="Emotion Detection Dashboard",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Real-Time Emotion Detection using CNN")

st.write("Detect human emotions from a live webcam feed.")

# ===========================
# LOAD MODEL
# ===========================

model = load_model("model/emotion_model.keras", compile=False)

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# ===========================
# LOAD HAARCASCADE
# ===========================

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("Unable to load Haar Cascade.")
    st.stop()

# ===========================
# SESSION STATE
# ===========================

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []

emotion_buffer = deque(maxlen=5)

# ===========================
# BUTTONS
# ===========================

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Camera"):
        st.session_state.camera_running = True

with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.camera_running = False

frame_placeholder = st.empty()

emotion_metric = st.empty()

confidence_metric = st.empty()

# ===========================
# CAMERA
# ===========================

if st.session_state.camera_running:

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("Unable to access webcam.")
        st.stop()

    while st.session_state.camera_running:

        success, frame = cap.read()

        if not success:
            st.error("Unable to read webcam.")
            break

        frame = cv2.resize(frame, (700, 500))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60,60)
        )

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (48,48))

            face = face.astype("float32") / 255.0

            face = np.expand_dims(face, axis=-1)

            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)

            confidence = float(np.max(prediction))

            emotion_index = int(np.argmax(prediction))

            emotion_buffer.append(emotion_index)

            emotion_index = max(
                set(emotion_buffer),
                key=emotion_buffer.count
            )

            emotion = emotion_labels[emotion_index]

            emotion_metric.metric(
                "Detected Emotion",
                emotion.capitalize()
            )

            confidence_metric.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"{emotion} ({confidence*100:.1f}%)",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            st.session_state.emotion_log.append({

                "Time": datetime.now().strftime("%H:%M:%S"),

                "Emotion": emotion,

                "Confidence": round(confidence*100,2)

            })

        frame_placeholder.image(
            frame,
            channels="BGR",
            use_container_width=True
        )

        time.sleep(0.02)

    cap.release()
# ===========================
# ANALYTICS SECTION
# ===========================

if len(st.session_state.emotion_log) > 0:

    st.markdown("---")

    st.subheader("📊 Emotion Detection History")

    df = pd.DataFrame(st.session_state.emotion_log)

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("📈 Emotion Distribution")

    emotion_count = df["Emotion"].value_counts()

    st.bar_chart(emotion_count)

    st.subheader("📋 Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Detections",
            len(df)
        )

    with col2:
        st.metric(
            "Most Frequent Emotion",
            emotion_count.idxmax()
        )

    with col3:
        st.metric(
            "Unique Emotions",
            df["Emotion"].nunique()
        )

    st.subheader("📥 Download Results")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV Report",
        data=csv,
        file_name="emotion_report.csv",
        mime="text/csv"
    )

    if st.button("🗑 Clear History"):

        st.session_state.emotion_log = []

        st.rerun()

else:

    st.info("No emotion data available. Start the camera to begin detection.")
