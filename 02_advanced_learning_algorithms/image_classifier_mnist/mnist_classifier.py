# =========================
# 1. Imports
# =========================
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

# =========================
# 2. Load Dataset (MNIST)
# =========================
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# 3. Build Model (Softmax classifier)
# =========================
model = Sequential([
    Flatten(input_shape=(28, 28)),      # image → vector
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')     # 🔥 softmax layer (10 digits)
])

# =========================
# 4. Compile Model
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 5. Train Model
# =========================
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.1,
    verbose=1
)

# =========================
# 6. Evaluate Model
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\nTest accuracy:", test_acc)

# =========================
# 7. Predictions
# =========================
predictions = model.predict(X_test)

# =========================
# 8. Helper: Show image + prediction
# =========================
def show_prediction(index):
    plt.imshow(X_test[index], cmap='gray')

    pred_label = np.argmax(predictions[index])
    confidence = np.max(predictions[index])

    plt.title(f"Predicted: {pred_label} | True: {y_test[index]} | Confidence: {confidence:.2f}")
    plt.axis('off')
    plt.show()

# Show a few examples
for i in range(5):
    show_prediction(i)

# =========================
# 9. Helper: Show probability distribution
# =========================
def show_probabilities(index):
    probs = predictions[index]

    plt.bar(range(10), probs)
    plt.title(f"Softmax probabilities for sample {index}")
    plt.xlabel("Digit class")
    plt.ylabel("Probability")
    plt.show()

# Example probability visualization
show_probabilities(0)

# =========================
# 10. Quick Accuracy Check (manual)
# =========================
pred_labels = np.argmax(predictions, axis=1)

accuracy = np.mean(pred_labels == y_test)
print("\nManual accuracy check:", accuracy)