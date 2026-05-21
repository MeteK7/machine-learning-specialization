# EMNIST OCR Camera
### From MNIST digit classifier → live camera character reader

---

## What's new vs. your MNIST project

| Feature | mnist_classifier | emnist_ocr |
|---|---|---|
| Classes | 10 (digits) | **62** (0–9, A–Z, a–z) |
| Architecture | Dense MLP | **CNN** (Conv + BN + Pool) |
| Input | static test set | **live webcam** |
| UI | matplotlib canvas | **OpenCV + tkinter** |
| Output | console only | **text buffer + save file** |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model  (~10 min GPU / ~60 min CPU)
#    First run also downloads the EMNIST dataset (~500 MB)
python 1_train_model.py

# 3. Launch the camera app  (requires a webcam)
python 2_camera_ocr.py
```

---

## How to use

### Mode 0 — DRAW MODE (yellow box)
1. Draw a single character inside the yellow box with your mouse
2. Press **SPACE** to predict it
3. Press **A** to append it to the text buffer
4. Press **C** to clear the canvas and draw the next character

### Mode 1 — CONTOUR MODE (green boxes)
1. Hold a piece of paper or whiteboard with **dark text on white background** up to the camera
2. The app auto-detects and predicts each character in real time
3. Press **SPACE** to capture the visible characters into the text buffer
4. Works best with clearly spaced, uppercase letters

### Text buffer (always visible at the bottom)
| Key | Action |
|-----|--------|
| `A` | Add current prediction to buffer |
| `BACKSPACE` | Delete last character |
| `E` | Open edit dialog |
| `S` | Open save-file dialog |
| `TAB` | Switch between DRAW / CONTOUR mode |
| `Q` / `ESC` | Quit |

---

## Tips for best accuracy

**Draw mode**
- Draw characters large and centered in the box
- Use thick strokes — the model was trained on thick pen-style characters
- The model expects **light character on dark background** (like chalk on blackboard); the app handles this automatically

**Contour mode**
- Use a clean white sheet of paper with a dark pen
- Write characters large (3–5 cm tall)
- Keep the paper flat and avoid shadows
- Uppercase letters tend to be more reliable than lowercase

---

## Architecture

```
Input (28×28×1)
  │
  ├── Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
  ├── Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
  ├── Conv2D(128) → BN → Conv2D(128) → MaxPool → Dropout(0.25)
  │
  ├── Flatten
  ├── Dense(512) → BN → Dropout(0.5)
  └── Dense(62, softmax)
```

Expected test accuracy: **~85–88%** on EMNIST/byclass

---

## File structure

```
emnist_ocr/
├── 1_train_model.py      ← train + save CNN
├── 2_camera_ocr.py       ← camera app
├── requirements.txt
├── README.md
├── emnist_model.keras     ← created after training
└── training_curves.png    ← created after training
```
