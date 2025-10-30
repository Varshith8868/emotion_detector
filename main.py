# main.py
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from utils import draw_radar_chart, overlay_image
from audio import speak_emotion
from history import EmotionHistory

# --- Config ---
MODEL_PATH = os.path.join("model", "emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load model
model = load_model(MODEL_PATH, compile=False)

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))

# History manager
history = EmotionHistory(labels=emotion_labels, maxlen=30)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera index.")

# Toggles
sound_enabled = True
show_history_panel = True

print("Press 's' to toggle sound, 'h' to toggle history panel, 'q' to quit.")

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make window fullscreen
    cv2.namedWindow('Emotion Detector (Modular)', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Emotion Detector (Modular)', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Mirror for natural feel
    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = face_detection.process(rgb)

    if detections.detections:
        for det in detections.detections:
            bboxC = det.location_data.relative_bounding_box
            x = int(bboxC.xmin * w_frame)
            y = int(bboxC.ymin * h_frame)
            w = int(bboxC.width * w_frame)
            h = int(bboxC.height * h_frame)
            x, y = max(0, x), max(0, y)
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)

            face = frame[y:y + h, x:x + w]
            if face.size == 0:
                continue

            try:
                roi = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)

                preds = model.predict(roi, verbose=0)[0]
                idx = int(np.argmax(preds))
                label = emotion_labels[idx]
                conf = float(np.max(preds))

                # Update history
                history.push(label, conf)

                # Speak emotion (optional)
                if sound_enabled:
                    speak_emotion(label)

                # Draw label
                cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Draw face mesh
                mesh_results = face_mesh.process(rgb)
                if mesh_results.multi_face_landmarks:
                    for fl in mesh_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=fl,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec
                        )

                # === Smaller Radar Chart (Top-Left) ===
                radar = draw_radar_chart(preds, emotion_labels, size=(150, 150))
                frame = overlay_image(frame, radar, 10, 10)

                # === Smaller History Bar Chart (Bottom-Left) ===
                if show_history_panel:
                    trend = history.trend_image(size=(250, 100))
                    frame = overlay_image(frame, trend, 10, h_frame - trend.shape[0] - 10)

            except Exception as e:
                print("Prediction error:", e)

    # Display final frame
    cv2.imshow('Emotion Detector (Modular)', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        sound_enabled = not sound_enabled
        print("Sound:", "ON" if sound_enabled else "OFF")
    elif key == ord('h'):
        show_history_panel = not show_history_panel
        print("History panel:", "ON" if show_history_panel else "OFF")

# Cleanup
cap.release()
cv2.destroyAllWindows()
