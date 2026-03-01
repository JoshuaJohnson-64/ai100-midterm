"""
CIFAR-10 Image Classification using a CNN
AI 100 - Midterm Project

No dataset download needed — CIFAR-10 loads automatically through Keras.

To run in Google Colab:
  1. Paste this entire file into a code cell
  2. Hit run. That's it.

Requirements (already in Colab by default):
  tensorflow, matplotlib, seaborn, scikit-learn, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Class Names ────────────────────────────────────────────────────────────────
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ── Load & Preprocess Data ────────────────────────────────────────────────────
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

print(f"Training samples : {x_train.shape[0]}")
print(f"Test samples     : {x_test.shape[0]}")
print(f"Image shape      : {x_train.shape[1:]}")

# ── Data Augmentation ─────────────────────────────────────────────────────────
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)

# ── Model Architecture ────────────────────────────────────────────────────────
def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_cnn()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting training...")
history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=64),
    epochs=30,
    validation_data=(x_test, y_test_cat),
    callbacks=callbacks
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

# ── Save Model ────────────────────────────────────────────────────────────────
model.save("cifar10_cnn_model.keras")
print("Model saved to cifar10_cnn_model.keras")

# ── Plots ─────────────────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/training_curves.png", dpi=150)
plt.show()

# Confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = y_test.flatten()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.show()

# Per-class report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Sample predictions visualization
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
indices = np.random.choice(len(x_test), 10, replace=False)
for i, idx in enumerate(indices):
    ax = axes[i // 5][i % 5]
    ax.imshow(x_test[idx])
    pred  = CLASS_NAMES[y_pred[idx]]
    truth = CLASS_NAMES[y_true[idx]]
    color = 'green' if pred == truth else 'red'
    ax.set_title(f"Pred: {pred}\nTrue: {truth}", color=color, fontsize=8)
    ax.axis('off')
plt.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=11)
plt.tight_layout()
plt.savefig("figures/sample_predictions.png", dpi=150)
plt.show()
