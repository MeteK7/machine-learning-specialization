# =============================================================================
# EMNIST OCR Camera App
# =============================================================================
# Two capture modes:
#   MODE 0 — DRAW   : draw a single character in the overlay box
#   MODE 1 — CONTOUR: auto-detect characters written on paper / whiteboard
#
# Keys (always active):
#   TAB      Switch between DRAW and CONTOUR mode
#   SPACE    Predict (draw mode) / Capture visible chars (contour mode)
#   A        Append last prediction to text buffer
#   BKSP     Delete last character from text buffer
#   E        Open edit dialog (tkinter)
#   S        Open save dialog  (tkinter)
#   C        Clear canvas / reset contour highlights
#   Q / ESC  Quit
# =============================================================================

import os
import sys
import threading
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import simpledialog, filedialog
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = "emnist_model.keras"
NUM_CLASSES  = 62
WIN_NAME     = "EMNIST OCR  |  TAB=switch mode  SPC=predict  A=add  E=edit  S=save  Q=quit"
CAMERA_INDEX = 0        # change to 1, 2 … if your webcam isn't at index 0
FRAME_W      = 1280
FRAME_H      = 720

# Contour detection sensitivity (tweak if too many / too few boxes appear)
CONTOUR_MIN_AREA   = 400    # px² — ignore tiny noise
CONTOUR_MAX_AREA   = 40_000 # px² — ignore large blobs (whole hand, etc.)
CONTOUR_MAX_BOXES  = 20     # cap predictions per frame

# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAP  (must match training script)
# ─────────────────────────────────────────────────────────────────────────────
def label_to_char(label: int) -> str:
    if label < 10:   return str(label)
    elif label < 36: return chr(label - 10 + ord('A'))
    else:            return chr(label - 36 + ord('a'))

LABEL_CHARS = [label_to_char(i) for i in range(NUM_CLASSES)]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] Model file not found: '{MODEL_PATH}'")
    print("Please run  1_train_model.py  first.\n")
    sys.exit(1)

print("Loading EMNIST model…", end=' ', flush=True)
model = tf.keras.models.load_model(MODEL_PATH)
print("done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def predict_char(roi_gray: np.ndarray) -> tuple[str, float]:
    """
    Predict a single character from a grayscale ROI (any size).
    Returns (char, confidence).
    """
    img = cv2.resize(roi_gray, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    # EMNIST is light-on-dark; invert if the image looks dark-on-light
    if img.mean() > 0.5:
        img = 1.0 - img

    probs = model.predict(img.reshape(1, 28, 28, 1), verbose=0)[0]
    idx   = int(np.argmax(probs))
    return LABEL_CHARS[idx], float(probs[idx])


# ─────────────────────────────────────────────────────────────────────────────
# DRAW CANVAS  (Mode 0)
# ─────────────────────────────────────────────────────────────────────────────
class DrawCanvas:
    """White drawing box overlaid on the camera feed (top-left corner)."""

    SIZE = 280      # canvas width & height in pixels
    PAD  = 30       # offset from frame corner
    BRUSH = 10      # brush radius

    def __init__(self):
        self.pixels  = np.ones((self.SIZE, self.SIZE), dtype=np.float32)
        self.drawing = False
        self.last    = ("?", 0.0)   # (char, confidence)

    def reset(self):
        self.pixels[:] = 1.0
        self.last = ("?", 0.0)

    def paint(self, cx: int, cy: int):
        """Draw a circle of black at canvas coords (cx, cy)."""
        cv2.circle(self.pixels, (cx, cy), self.BRUSH, 0.0, -1)

    def predict(self) -> tuple[str, float]:
        roi = (self.pixels * 255).astype(np.uint8)
        self.last = predict_char(roi)
        return self.last

    def frame_to_canvas(self, fx: int, fy: int) -> tuple[int, int] | None:
        """Convert frame pixel → canvas pixel; None if outside the box."""
        cx = fx - self.PAD
        cy = fy - self.PAD
        if 0 <= cx < self.SIZE and 0 <= cy < self.SIZE:
            return cx, cy
        return None

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """Blend the canvas into the frame and annotate it."""
        x0, y0 = self.PAD, self.PAD
        x1, y1 = x0 + self.SIZE, y0 + self.SIZE

        # blend: 80% canvas, 20% camera feed
        canvas_bgr = cv2.cvtColor(
            (self.pixels * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        region = frame[y0:y1, x0:x1]
        frame[y0:y1, x0:x1] = cv2.addWeighted(canvas_bgr, 0.80, region, 0.20, 0)

        # box border
        cv2.rectangle(frame, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2),
                      (255, 220, 60), 2)
        # header
        cv2.putText(frame, "DRAW MODE  — draw one char, press SPACE",
                    (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 220, 60), 1, cv2.LINE_AA)

        # prediction badge below the box
        char, conf = self.last
        if char != "?":
            badge = f"  '{char}'   {conf:.1%}"
            cv2.putText(frame, badge, (x0, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 120), 2, cv2.LINE_AA)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# CONTOUR DETECTOR  (Mode 1)
# ─────────────────────────────────────────────────────────────────────────────
def find_characters(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Find bounding boxes of characters via adaptive thresholding + contours.
    Best results with dark text on light background (paper / whiteboard).
    Returns list of (x, y, w, h) sorted left-to-right.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (CONTOUR_MIN_AREA < area < CONTOUR_MAX_AREA):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if 0.08 < aspect < 3.5:       # filter elongated noise
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])    # left → right
    return boxes[:CONTOUR_MAX_BOXES]


# ─────────────────────────────────────────────────────────────────────────────
# TEXT BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class TextBuffer:
    def __init__(self):
        self._chars: list[str] = []

    def add(self, char: str):
        self._chars.append(char)

    def add_string(self, s: str):
        self._chars.extend(list(s))

    def backspace(self):
        if self._chars:
            self._chars.pop()

    def get(self) -> str:
        return ''.join(self._chars)

    def set(self, s: str):
        self._chars = list(s)

    def clear(self):
        self._chars.clear()


# ─────────────────────────────────────────────────────────────────────────────
# TKINTER DIALOGS  (run blocking from main thread — pauses camera loop)
# ─────────────────────────────────────────────────────────────────────────────
def _make_root() -> tk.Tk:
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    return root

def dialog_edit(buf: TextBuffer):
    """Open a text-edit dialog; updates buf in-place."""
    root = _make_root()
    result = simpledialog.askstring(
        "Edit Recognized Text",
        "Edit the text below, then click OK:",
        initialvalue=buf.get(),
        parent=root
    )
    root.destroy()
    if result is not None:
        buf.set(result)
        print(f"[Buffer] Edited → '{buf.get()}'")

def dialog_save(buf: TextBuffer):
    """Open a save-file dialog."""
    if not buf.get():
        print("[Save] Buffer is empty — nothing to save.")
        return

    root = _make_root()
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = filedialog.asksaveasfilename(
        title="Save Recognized Text",
        defaultextension=".txt",
        initialfile=f"ocr_{ts}.txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        parent=root
    )
    root.destroy()
    if path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(buf.get())
        print(f"[Saved] → {path}")
    else:
        print("[Save] Cancelled.")


# ─────────────────────────────────────────────────────────────────────────────
# FRAME OVERLAYS
# ─────────────────────────────────────────────────────────────────────────────
def draw_text_banner(frame: np.ndarray, buf: TextBuffer):
    """Semi-transparent banner at the bottom showing the accumulated text."""
    h, w = frame.shape[:2]
    bh = 52

    # dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bh), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    text = buf.get()
    # truncate from the left if too long
    max_chars = 70
    if len(text) > max_chars:
        text = '…' + text[-max_chars:]

    label  = f"Text: {text if text else '<empty>'}"
    cursor = "|" if (int(cv2.getTickCount() / cv2.getTickFrequency() * 2) % 2) else ""
    cv2.putText(frame, label + cursor,
                (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (180, 255, 180), 2, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, mode: int):
    """Key-binding cheat-sheet in the top-right corner."""
    h, w = frame.shape[:2]

    lines_draw    = ["── DRAW MODE ──", "TAB  Switch mode",
                     "SPC  Predict", "A    Add to text",
                     "BKSP Delete last", "C    Clear canvas",
                     "E    Edit text", "S    Save text", "Q    Quit"]
    lines_contour = ["── CONTOUR MODE ──", "TAB  Switch mode",
                     "SPC  Capture line", "A    Add last char",
                     "BKSP Delete last", "C    Reset",
                     "E    Edit text", "S    Save text", "Q    Quit"]

    lines   = lines_draw if mode == 0 else lines_contour
    color_h = (80, 220, 255) if mode == 0 else (80, 180, 255)

    for i, ln in enumerate(lines):
        color = color_h if i == 0 else (190, 190, 190)
        cv2.putText(frame, ln,
                    (w - 230, 28 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera (index {CAMERA_INDEX}).")
        print("Try changing CAMERA_INDEX at the top of this file.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, FRAME_W, FRAME_H)

    # ── State ────────────────────────────────────────────────────────────────
    mode    = 0                # 0 = DRAW, 1 = CONTOUR
    canvas  = DrawCanvas()
    buf     = TextBuffer()

    # contour mode state
    c_boxes : list[tuple] = []          # (x, y, w, h, char, conf)
    c_string: str         = ""          # chars from last frame

    # ── Mouse callback (draw mode only) ──────────────────────────────────────
    def on_mouse(event, fx, fy, flags, _):
        if mode != 0:
            return
        coords = canvas.frame_to_canvas(fx, fy)
        if coords is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            canvas.drawing = True
            canvas.paint(*coords)
        elif event == cv2.EVENT_MOUSEMOVE and canvas.drawing:
            canvas.paint(*coords)
        elif event == cv2.EVENT_LBUTTONUP:
            canvas.drawing = False

    cv2.setMouseCallback(WIN_NAME, on_mouse)

    print("─" * 60)
    print("  EMNIST OCR Camera  —  running")
    print("─" * 60)
    print("  TAB   : switch mode (DRAW ↔ CONTOUR)")
    print("  SPACE : predict (draw) / capture line (contour)")
    print("  A     : append prediction to text buffer")
    print("  BKSP  : delete last character")
    print("  E     : edit text (dialog)")
    print("  S     : save text (dialog)")
    print("  C     : clear canvas")
    print("  Q/ESC : quit")
    print("─" * 60, "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame — retrying…")
            continue

        frame = cv2.flip(frame, 1)                         # mirror
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── MODE 0 : DRAW ─────────────────────────────────────────────────
        if mode == 0:
            frame = canvas.overlay(frame)

        # ── MODE 1 : CONTOUR ──────────────────────────────────────────────
        else:
            boxes = find_characters(gray)
            c_boxes = []
            for (x, y, w, h) in boxes:
                pad  = 6
                y0c  = max(0,            y - pad)
                y1c  = min(gray.shape[0], y + h + pad)
                x0c  = max(0,            x - pad)
                x1c  = min(gray.shape[1], x + w + pad)
                roi  = gray[y0c:y1c, x0c:x1c]
                if roi.size == 0:
                    continue
                char, conf = predict_char(roi)
                c_boxes.append((x, y, w, h, char, conf))

                # bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (60, 210, 60), 2)
                # label above the box
                lbl = f"{char}  {conf:.0%}"
                cv2.putText(frame, lbl,
                            (x, max(y - 6, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 100), 2, cv2.LINE_AA)

            c_string = ''.join(r[4] for r in c_boxes)

            # show the assembled string in the middle of the frame
            if c_string:
                fw = frame.shape[1]
                cv2.putText(frame, f"Reading: {c_string}",
                            (20, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (0, 220, 255), 2, cv2.LINE_AA)

        # ── Common overlays ───────────────────────────────────────────────
        draw_text_banner(frame, buf)
        draw_hud(frame, mode)

        cv2.imshow(WIN_NAME, frame)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        # ── QUIT ──────────────────────────────
        if key in (ord('q'), 27):        # Q or ESC
            break

        # ── SWITCH MODE ───────────────────────
        elif key == 9:                   # TAB
            mode = 1 - mode
            canvas.reset()
            c_boxes  = []
            c_string = ""
            print(f"[Mode] → {'CONTOUR' if mode else 'DRAW'}")

        # ── CLEAR ─────────────────────────────
        elif key == ord('c'):
            canvas.reset()
            c_boxes  = []
            c_string = ""

        # ── PREDICT (draw) / CAPTURE (contour) ─
        elif key == ord(' '):
            if mode == 0:
                char, conf = canvas.predict()
                print(f"[DRAW]    Predicted: '{char}'  ({conf:.1%})")
            else:
                if c_string:
                    buf.add_string(c_string + ' ')
                    print(f"[CONTOUR] Captured: '{c_string}'  →  '{buf.get()}'")
                else:
                    print("[CONTOUR] No characters detected.")

        # ── ADD LAST PREDICTION → BUFFER ──────
        elif key == ord('a'):
            if mode == 0:
                char, conf = canvas.last
                if char != "?":
                    buf.add(char)
                    print(f"[Buffer] +'{char}'  →  '{buf.get()}'")
                else:
                    print("[Buffer] Nothing to add — press SPACE first.")
            else:
                # add the last contour string character by character
                if c_string:
                    buf.add_string(c_string)
                    print(f"[Buffer] +'{c_string}'  →  '{buf.get()}'")

        # ── BACKSPACE ─────────────────────────
        elif key in (8, 127):            # Backspace / Delete
            buf.backspace()
            print(f"[Buffer] ← '{buf.get()}'")

        # ── EDIT ──────────────────────────────
        elif key == ord('e'):
            dialog_edit(buf)

        # ── SAVE ──────────────────────────────
        elif key == ord('s'):
            dialog_save(buf)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("\n[Exited]  Final buffer contents:")
    print(f"  '{buf.get()}'\n")


if __name__ == "__main__":
    main()
