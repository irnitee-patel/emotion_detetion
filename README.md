# 😊 Real-Time Emotion Detection Using CNN

## 📌 Project Overview

This project detects human facial emotions in real time using a Convolutional Neural Network (CNN), OpenCV, and Streamlit. The application captures live video from a webcam, detects faces, predicts emotions, and displays the detected emotion with its confidence score on a user-friendly dashboard.

---

## 🎯 Features

* Real-time webcam emotion detection
* Face detection using Haar Cascade Classifier
* CNN-based emotion classification
* Detects seven emotions:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Sad
  * Surprise
* Displays prediction confidence
* Emotion history table
* Emotion distribution chart
* CSV report download
* Clean and interactive Streamlit dashboard

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📂 Project Structure

```
Emotion_Detection/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── model/
│   └── emotion_model.keras
│
├── train_model.py
├── dashboard.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```
git clone <repository-link>
```

Move into the project folder:

```
cd Emotion_Detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶ Training the Model

Run:

```
python train_model.py
```

The trained model will be saved inside the **model** folder.

---

## ▶ Running the Dashboard

Launch the application:

```
streamlit run dashboard.py
```

The dashboard will open automatically in your web browser.

---

## 📊 Dashboard Features

* Live webcam feed
* Face detection
* Emotion prediction
* Confidence score
* Emotion history
* Emotion distribution visualization
* CSV export

---

## 📈 Model Information

* Image Size: 48 × 48
* Color Mode: Grayscale
* CNN Architecture
* Adam Optimizer
* Categorical Crossentropy Loss
* Early Stopping
* Reduce Learning Rate on Plateau
* Model Checkpoint

---

## 📚 Future Improvements

* MediaPipe Face Detection
* Deep learning models such as ResNet or MobileNetV2
* Multiple face tracking
* Emotion trend analysis
* Cloud deployment
* REST API integration

---

## 👨‍💻 Author

Irnitee C. Patel

B.Tech Computer Engineering (Artificial Intelligence)
