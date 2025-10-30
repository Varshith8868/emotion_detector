# 🎭 Emotion Detection from Facial Expressions

This project detects **human emotions in real-time** using a webcam feed.  
It leverages **Deep Learning (CNN)** with **TensorFlow/Keras**, and **MediaPipe** for face detection.  
Emotions are visualized using a **radar graph**, and a voice announces each detected emotion.  
Recent detections are tracked in a **history chart** for better insight.

---

## 🚀 Features

- 🧠 Real-time facial emotion recognition (`Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`)
- 🕸️ Face mesh overlay using **MediaPipe**
- 📊 Live radar chart showing emotion probabilities
- 🔊 Voice feedback (emotion is spoken aloud)
- 🗃️ Emotion history tracking (saved in `logs/emotion_log.csv`)
- 🎛️ Simple UI toggles:
  - `S` → Toggle sound  ON/OFF
  - `H` → Toggle history panel  ON/OFF
  - `Q` → Quit  
---

## 🧩 Tech Stack

- **Python 3.x**
- **TensorFlow / Keras** – Emotion classification model  
- **MediaPipe** – Face detection & mesh drawing  
- **OpenCV** – Real-time video processing  
- **Matplotlib** – Radar & history visualizations  
- **pyttsx3 / playsound** – Voice output  

---

## 📦 Installation

1. Clone or download this repository  
2. Open terminal in the project folder  
3. Create and activate a virtual environment  
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
