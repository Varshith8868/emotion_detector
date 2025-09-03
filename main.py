import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
model = load_model ('model/emotion_model.h5', compile=False)

# Emotion labels based on FER-2013
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
# Load OpenCV's Haar cascade face detector
face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Start webcam 
cap=cv2.VideoCapture(0) 
while True:
    ret,frame= cap.read() 
    if not ret :  
        break;
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces :
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y),(x+w, y+h),(255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64),interpolation=cv2.INTER_AREA)

        # Preprocess ROI and predict
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = np.expand_dims(roi,axis=-1)      # Add channel dimension → (64, 64, 1)
            roi = np.expand_dims(roi,axis=0)        # Add batch dimension → (1, 64, 64, 1)

            prediction = model.predict(roi, verbose=0)[0]
            label=emotion_labels[np.argmax(prediction)]

            label_position=(x,y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey (1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
