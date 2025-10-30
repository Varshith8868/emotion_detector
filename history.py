# history.py
import csv
import os
from collections import deque
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
CSV_PATH = os.path.join(LOGS_DIR, "emotion_log.csv")

class EmotionHistory:
    def __init__(self, labels, maxlen=30):
        self.labels = labels
        self.maxlen = maxlen
        self.deque = deque(maxlen=maxlen)
        # ensure header in csv
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "emotion", "confidence"])

    def push(self, emotion, confidence):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.deque.append(emotion)
        # append to CSV
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, emotion, f"{confidence:.4f}"])

    def get_recent(self):
        return list(self.deque)

    def get_counts(self):
        counts = {lab: 0 for lab in self.labels}
        for e in self.deque:
            counts[e] = counts.get(e, 0) + 1
        return counts

    def trend_image(self, size=(300, 150)):
        """
        Returns a small chart image (BGR) showing counts of recent emotions
        """
        counts = self.get_counts()
        labels = list(counts.keys())
        vals = [counts[l] for l in labels]

        fig, ax = plt.subplots(figsize=(3,1.5))
        ax.bar(labels, vals)
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_ylabel("Count")
        ax.set_title("Recent Emotion Counts")
        plt.tight_layout()

        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf).convert("RGBA")
        img = img.resize(size)
        arr = np.array(img)
        buf.close()
        plt.close(fig)
        # RGBA -> BGR
        import cv2
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
