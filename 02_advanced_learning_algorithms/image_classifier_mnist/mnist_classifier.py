# =========================
# 1. Imports
# =========================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

from PIL import Image

# =========================
# 2. Load Dataset (MNIST)
# =========================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0–255 → 0–1)
X_train = X_train / 255.0
X_test  = X_test  / 255.0

print("Train shape:", X_train.shape)
print("Test shape: ", X_test.shape)

# =========================
# 3. Build / Load Model
# =========================
MODEL_PATH = "mnist_model.keras"

if os.path.exists(MODEL_PATH):
    print("\nLoading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded — skipping training.")
else:
    # -------------------------
    # Build
    # -------------------------
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64,  activation='relu'),
        Dense(10,  activation='softmax')
    ])

    # -------------------------
    # Compile
    # -------------------------
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # -------------------------
    # Train
    # -------------------------
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )

    # -------------------------
    # Save
    # -------------------------
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

# =========================
# 4. Evaluate Model
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\nTest accuracy:", test_acc)

# =========================
# 5. Predictions on MNIST
# =========================
predictions = model.predict(X_test, verbose=0)

# =========================
# 6. Manual Accuracy Check
# =========================
pred_labels = np.argmax(predictions, axis=1)
accuracy    = np.mean(pred_labels == y_test)
print("Manual accuracy check:", accuracy)

# =========================
# 7. Show Prediction
# =========================
def show_prediction(index):

    plt.imshow(X_test[index], cmap='gray')

    pred_label = np.argmax(predictions[index])
    confidence = np.max(predictions[index])

    plt.title(
        f"Predicted: {pred_label} | "
        f"True: {y_test[index]} | "
        f"Confidence: {confidence:.2f}"
    )

    plt.axis('off')
    plt.show()

# =========================
# 8. Show Probabilities
# =========================
def show_probabilities(index):

    probs = predictions[index]

    plt.bar(range(10), probs)
    plt.title(f"Softmax probabilities for sample {index}")
    plt.xlabel("Digit class")
    plt.ylabel("Probability")
    plt.show()

# =========================
# 9. DRAWING UI
# =========================
def draw_digit():

    # Large canvas for easier drawing (white background = 1.0)
    canvas = np.ones((280, 280))

    fig, ax = plt.subplots()
    ax.set_title("Draw a digit with your mouse.\nClose the window when finished.")

    image_display = ax.imshow(
        canvas,
        cmap='gray',
        vmin=0,
        vmax=1
    )

    plt.axis('off')

    drawing = False

    # -------------------------
    # Mouse Press
    # -------------------------
    def on_press(event):
        nonlocal drawing
        drawing = True

    # -------------------------
    # Mouse Release
    # -------------------------
    def on_release(event):
        nonlocal drawing
        drawing = False

    # -------------------------
    # Mouse Movement
    # -------------------------
    def on_move(event):

        if drawing and event.xdata is not None and event.ydata is not None:

            x = int(event.xdata)
            y = int(event.ydata)

            if 0 <= x < 280 and 0 <= y < 280:

                # FIX: clamp brush bounds so negative indices are never used
                y0, y1 = max(0, y - 6), min(280, y + 6)
                x0, x1 = max(0, x - 6), min(280, x + 6)

                canvas[y0:y1, x0:x1] = 0  # draw black pixels

                image_display.set_data(canvas)
                fig.canvas.draw_idle()

    # Connect mouse events
    fig.canvas.mpl_connect('button_press_event',   on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event',  on_move)

    plt.show()

    # =========================
    # Convert to 28×28
    # =========================
    img = Image.fromarray((canvas * 255).astype(np.uint8))

    # FIX: use LANCZOS for consistent, high-quality downscaling
    img = img.resize((28, 28), Image.LANCZOS)

    img_array = np.array(img) / 255.0

    # FIX: invert so the digit is white-on-black, matching MNIST format
    img_array = 1.0 - img_array

    return img_array

# =========================
# 10. Predict Drawn Digit
# =========================
def predict_custom_digit(img):

    img_input  = img.reshape(1, 28, 28)

    # FIX: verbose=0 suppresses the distracting progress bar
    prediction = model.predict(img_input, verbose=0)

    pred_label = np.argmax(prediction)
    confidence = np.max(prediction)

    print("\nPredicted Digit:", pred_label)
    print("Confidence:     ", round(confidence, 4))

    # Show image
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {pred_label} | Confidence: {confidence:.2f}")
    plt.axis('off')
    plt.show()

    # Probability graph
    plt.bar(range(10), prediction[0])
    plt.title("Softmax Probabilities")
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.show()

# =========================
# 11. MAIN MENU
# =========================
while True:

    print("\n=========================")
    print("MNIST DIGIT AI")
    print("=========================")
    print("1 -> Test MNIST Images")
    print("2 -> Draw Your Own Digit")
    print("3 -> Exit")

    choice = input("\nSelect option: ").strip()

    # =========================
    # Option 1
    # =========================
    if choice == "1":

        # FIX: validate input — catch non-integers and out-of-range values
        try:
            index = int(input(f"Enter test image index (0-{len(X_test)-1}): "))
            if not 0 <= index < len(X_test):
                raise ValueError
        except ValueError:
            print(f"\nInvalid input. Please enter a number between 0 and {len(X_test)-1}.")
            continue

        show_prediction(index)
        show_probabilities(index)

    # =========================
    # Option 2
    # =========================
    elif choice == "2":

        drawn_digit = draw_digit()
        predict_custom_digit(drawn_digit)

    # =========================
    # Option 3
    # =========================
    elif choice == "3":

        print("\nExiting...")
        break

    else:
        print("\nInvalid option. Please enter 1, 2, or 3.")