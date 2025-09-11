# ---Vesion 2---
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image

# Load the trained model
model = load_model('model/emotion_model.h5', compile=False)

# Emotion labels
emotion_labels = ['Angry','Disgust','Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MediaPipe components
mp_face_detection=mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_detection=mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)

# Drawing spec for face mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 255))

# Start webcam
cap = cv2.VideoCapture(0)

def draw_radar_chart(preds):
    # Create radar chart from predictions
    labels = emotion_labels
    values = preds.tolist()
    values += values[:1]  # close the radar chart

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='green', linewidth=2)
    ax.fill(angles, values, color='lime', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True)

    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    img = np.array(img)
    buf.close()
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural webcam view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Make sure bounding box is within the frame
            x, y = max(0, x), max(0, y)
            face_img = frame[y:y + h, x:x + w]

            # Emotion Prediction
            try:
                roi = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)

                preds = model.predict(roi, verbose=0)[0]
                label = emotion_labels[np.argmax(preds)]
                confidence = np.max(preds)

                # Draw label
                cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Radar chart beside face
                radar_img = draw_radar_chart(preds)
                radar_h, radar_w, _ = radar_img.shape

                # Resize radar chart
                if y + radar_h > frame.shape[0]:
                    radar_img = radar_img[:frame.shape[0] - y, :, :]
                if x + w + 10 + radar_w > frame.shape[1]:
                    radar_img = radar_img[:, :frame.shape[1] - (x + w + 10), :]

                frame[y:y + radar_img.shape[0], x + w + 10:x + w + 10 + radar_img.shape[1]] = radar_img

            except Exception as e:
                print("Emotion detection failed:", e)
                 # Detect and draw face mesh
    mesh_results = face_mesh.process(rgb_frame)
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Display the frame
    cv2.imshow('Emotion Detector + Face Mesh + Radar Graph', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()