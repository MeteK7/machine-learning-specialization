# =========================
# 1. Imports
# =========================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64,  activation='relu'),
        Dense(10,  activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

# =========================
# 4. Evaluate Model
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\nTest accuracy:", test_acc)

# =========================
# 5. Predictions on MNIST test set
# =========================
predictions = model.predict(X_test, verbose=0)

# =========================
# 6. Manual Accuracy Check
# =========================
pred_labels = np.argmax(predictions, axis=1)
accuracy    = np.mean(pred_labels == y_test)
print("Manual accuracy check:", accuracy)

# =========================
# 7. Show MNIST Test Prediction
# =========================
def show_prediction(index):
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("MNIST Test Sample", fontsize=13)

    ax_img.imshow(X_test[index], cmap='gray')
    pred_label = np.argmax(predictions[index])
    confidence = np.max(predictions[index])
    ax_img.set_title(
        f"Predicted: {pred_label}  |  True: {y_test[index]}\n"
        f"Confidence: {confidence:.2%}"
    )
    ax_img.axis('off')

    colors = ['tomato' if i == pred_label else 'steelblue' for i in range(10)]
    ax_bar.bar(range(10), predictions[index], color=colors)
    ax_bar.set_title("Softmax Probabilities")
    ax_bar.set_xlabel("Digit class")
    ax_bar.set_ylabel("Probability")
    ax_bar.set_xticks(range(10))
    ax_bar.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

# =========================
# 8. Drawing UI (Persistent Window)
# =========================
def draw_digit_and_predict():
    canvas = np.ones((280, 280))   # white background

    fig = plt.figure(figsize=(13, 5))
    fig.suptitle("MNIST Digit AI — Draw Your Digit", fontsize=13)
    plt.subplots_adjust(left=0.05, right=0.97, bottom=0.22, top=0.88, wspace=0.45)

    ax_draw = fig.add_subplot(1, 3, 1)
    ax_prev = fig.add_subplot(1, 3, 2)
    ax_prob = fig.add_subplot(1, 3, 3)

    ax_draw.set_title("Draw here", fontsize=11)
    img_display = ax_draw.imshow(canvas, cmap='gray', vmin=0, vmax=1)
    ax_draw.axis('off')

    ax_prev.set_title("Model Input Preview (28×28)", fontsize=11)
    prev_display = ax_prev.imshow(np.zeros((28, 28)), cmap='gray', vmin=0, vmax=1)
    ax_prev.axis('off')

    ax_prob.set_title("Prediction", fontsize=11)
    bars = ax_prob.bar(range(10), np.zeros(10), color='steelblue')
    ax_prob.set_xlim(-0.5, 9.5)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlabel("Digit")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_xticks(range(10))

    drawing = False
    def on_press(event):
        nonlocal drawing
        if event.inaxes == ax_draw:
            drawing = True
    def on_release(event):
        nonlocal drawing
        drawing = False
    def on_move(event):
        if drawing and event.inaxes == ax_draw and event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < 280 and 0 <= y < 280:
                y0, y1 = max(0, y - 6), min(280, y + 6)
                x0, x1 = max(0, x - 6), min(280, x + 6)
                canvas[y0:y1, x0:x1] = 0
                img_display.set_data(canvas)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event',   on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event',  on_move)

    ax_btn_predict = plt.axes([0.38, 0.05, 0.13, 0.09])
    ax_btn_clear   = plt.axes([0.53, 0.05, 0.13, 0.09])

    btn_predict = Button(ax_btn_predict, 'Predict', color='#d0e8ff', hovercolor='#a8d0ff')
    btn_clear   = Button(ax_btn_clear,   'Clear',   color='#ffe0d0', hovercolor='#ffb8a0')

    def on_predict(event):
        img = Image.fromarray((canvas * 255).astype(np.uint8))
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = 1.0 - img_array

        prev_display.set_data(img_array)
        prediction = model.predict(img_array.reshape(1, 28, 28), verbose=0)
        pred_label = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        for i, (bar, h) in enumerate(zip(bars, prediction[0])):
            bar.set_height(h)
            bar.set_color('tomato' if i == pred_label else 'steelblue')

        ax_prob.set_title(f"Predicted: {pred_label}   ({confidence:.1%})", fontsize=11, color='tomato')
        print(f"\nPredicted Digit : {pred_label}")
        print(f"Confidence      : {confidence:.4f}")
        fig.canvas.draw_idle()

    def on_clear(event):
        canvas[:] = 1.0
        img_display.set_data(canvas)
        prev_display.set_data(np.zeros((28, 28)))
        for bar in bars:
            bar.set_height(0)
            bar.set_color('steelblue')
        ax_prob.set_title("Prediction", fontsize=11, color='black')
        fig.canvas.draw_idle()

    btn_predict.on_clicked(on_predict)
    btn_clear.on_clicked(on_clear)

    plt.show()

# =========================
# 9. MAIN MENU
# =========================
while True:
    print("\n=========================")
    print("MNIST DIGIT AI")
    print("=========================")
    print("1 -> Test MNIST Images")
    print("2 -> Draw Your Own Digit")
    print("3 -> Exit")

    choice = input("\nSelect option: ").strip()

    if choice == "1":
        try:
            index = int(input(f"Enter test image index (0-{len(X_test)-1}): "))
            if not 0 <= index < len(X_test):
                raise ValueError
        except ValueError:
            print(f"\nInvalid input. Please enter a number between 0 and {len(X_test)-1}.")
            continue
        show_prediction(index)

    elif choice == "2":
        draw_digit_and_predict()

    elif choice == "3":
        print("\nExiting...")
        break

    else:
        print("\nInvalid option. Please enter 1, 2, or 3.")
