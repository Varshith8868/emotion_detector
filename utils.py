# utils.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image
import cv2

def draw_radar_chart(preds, labels, size=(300, 300)):
    """
    preds: 1D numpy array of probabilities for each label (normalized 0..1)
    labels: list of label names
    returns: BGR OpenCV image (H,W,3)
    """
    # Ensure normalized
    vals = preds.tolist()
    vals += vals[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True)

    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    img = img.resize(size)
    arr = np.array(img)
    buf.close()
    plt.close(fig)
    # Convert RGBA to BGR for OpenCV
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return bgr

def overlay_image(bg, fg, x, y):
    """
    Overlay fg image onto bg at (x, y). Both are numpy BGR arrays.
    Will clip if out of bounds.
    Returns modified bg (copy).
    """
    bh, bw = bg.shape[:2]
    fh, fw = fg.shape[:2]

    # Clip to fit
    if x >= bw or y >= bh:
        return bg
    w = min(fw, bw - x)
    h = min(fh, bh - y)
    if w <= 0 or h <= 0:
        return bg

    bg_section = bg[y:y+h, x:x+w]
    fg_section = fg[:h, :w]

    # If fg has alpha channel (4), handle it
    if fg_section.shape[2] == 4:
        alpha = fg_section[:, :, 3] / 255.0
        for c in range(3):
            bg_section[:, :, c] = (alpha * fg_section[:, :, c] +
                                   (1 - alpha) * bg_section[:, :, c])
    else:
        # Simple replace
        bg_section[:] = fg_section

    bg[y:y+h, x:x+w] = bg_section
    return bg
