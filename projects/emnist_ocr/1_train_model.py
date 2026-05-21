# =============================================================================
# EMNIST Character Classifier — Training Script
# =============================================================================
# Dataset : EMNIST/byclass  (62 classes: digits 0-9, A-Z, a-z)
# Model   : CNN  (Conv → BN → Pool → Dropout blocks, ~2.5M params)
# Output  : emnist_model.keras
#
# First run downloads ~500 MB. Subsequent runs load the saved model.
# Training time: ~10 min (GPU) / ~60 min (CPU).
# Expected test accuracy: ~85-88%
# =============================================================================

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib
matplotlib.use('Agg')          # headless-safe for saving plots
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten,
    Dropout, BatchNormalization
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = "emnist_model.keras"
NUM_CLASSES = 62           # 10 digits + 26 upper + 26 lower
EPOCHS      = 15
BATCH_SIZE  = 256
IMG_SHAPE   = (28, 28, 1)

# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────────────────────────────────────
def label_to_char(label: int) -> str:
    """
    EMNIST/byclass label ordering:
      0 – 9   →  '0'–'9'
     10 – 35  →  'A'–'Z'
     36 – 61  →  'a'–'z'
    """
    if label < 10:
        return str(label)
    elif label < 36:
        return chr(label - 10 + ord('A'))
    else:
        return chr(label - 36 + ord('a'))

LABEL_CHARS = [label_to_char(i) for i in range(NUM_CLASSES)]
print(f"62 classes: {''.join(LABEL_CHARS)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Loading EMNIST/byclass dataset (~500 MB, first run downloads it)...")
    (ds_train, ds_test), info = tfds.load(
        'emnist/byclass',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True
    )
    print(f"  Train samples : {info.splits['train'].num_examples:,}")
    print(f"  Test  samples : {info.splits['test'].num_examples:,}\n")

    def preprocess(image, label):
        # EMNIST images are stored transposed vs. standard orientation → fix it
        img = tf.cast(image, tf.float32) / 255.0   # normalize 0→1
        img = tf.squeeze(img, axis=-1)              # (28,28,1) → (28,28)
        img = tf.transpose(img)                     # fix 90° rotation artifact
        img = tf.expand_dims(img, axis=-1)          # (28,28) → (28,28,1)
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE

    ds_train = (ds_train
                .map(preprocess,  num_parallel_calls=AUTOTUNE)
                .cache()
                .shuffle(20_000)
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE))

    ds_test  = (ds_test
                .map(preprocess,  num_parallel_calls=AUTOTUNE)
                .cache()
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL — build or load
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    print(f"Saved model found at '{MODEL_PATH}' — loading (skipping training).")
    model = tf.keras.models.load_model(MODEL_PATH)

else:
    model = Sequential([
        # ── Block 1 ──────────────────────────────
        Conv2D(32, (3, 3), padding='same', activation='relu',
               input_shape=IMG_SHAPE),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Block 2 ──────────────────────────────
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Block 3 ──────────────────────────────
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Classifier head ──────────────────────
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=2, min_lr=1e-6, verbose=1
        ),
    ]

    print("\nTraining...\n")
    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_test,
        callbacks=callbacks
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # ── Training curves ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EMNIST CNN Training", fontsize=13)

    ax1.plot(history.history['accuracy'],     label='Train acc')
    ax1.plot(history.history['val_accuracy'], label='Val acc')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train loss')
    ax2.plot(history.history['val_loss'], label='Val loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved → training_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE  (always runs — loads dataset fresh if we skipped training above)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading test split for evaluation…")
ds_test_eval, _ = tfds.load(
    'emnist/byclass',
    split='test',
    as_supervised=True,
    with_info=True
)

def preprocess_eval(image, label):
    img = tf.cast(image, tf.float32) / 255.0
    img = tf.squeeze(img, axis=-1)
    img = tf.transpose(img)
    img = tf.expand_dims(img, axis=-1)
    return img, label

ds_test_eval = (ds_test_eval
                .map(preprocess_eval, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

test_loss, test_acc = model.evaluate(ds_test_eval, verbose=1)
print(f"\n{'='*40}")
print(f"  Test accuracy : {test_acc:.4f}  ({test_acc:.2%})")
print(f"  Test loss     : {test_loss:.4f}")
print(f"{'='*40}")
print("\nDone! You can now run  2_camera_ocr.py")
