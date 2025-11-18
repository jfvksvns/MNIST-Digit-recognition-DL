# train_mnist_model.py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Create model directory if it doesn't exist
os.makedirs("../model", exist_ok=True)

# 1. Load MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)   # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# 3. One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 4. Data Augmentation (helps model generalize better to hand-drawn digits)
datagen = ImageDataGenerator(
    rotation_range=10,          # Randomly rotate images by up to 10 degrees
    width_shift_range=0.1,      # Randomly shift images horizontally by 10%
    height_shift_range=0.1,     # Randomly shift images vertically by 10%
    zoom_range=0.1,             # Randomly zoom images by 10%
    shear_range=0.1             # Apply shear transformations
)
datagen.fit(x_train)

# 5. Define Enhanced CNN model
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1), padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    # Fully Connected Layers
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

# 6. Compile model with better optimizer settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Print model summary
print("\nModel Architecture:")
model.summary()

# 7. Setup callbacks for better training
callbacks = [
    # Early stopping: stop training if validation loss doesn't improve
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        '../model/mnist_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 8. Train model with data augmentation
print("\nStarting training...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=30,  # Increased epochs with early stopping
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 9. Evaluate on test set
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# 10. Make predictions on test set to show per-digit accuracy
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate per-digit accuracy
print("\nPer-digit accuracy:")
for digit in range(10):
    digit_mask = y_true_classes == digit
    digit_acc = np.mean(y_pred_classes[digit_mask] == digit)
    print(f"Digit {digit}: {digit_acc:.4f}")

# 11. Save final model
model.save("../model/mnist_model.h5")
print("\nFinal model saved to ../model/mnist_model.h5")
print("Best model saved to ../model/mnist_model_best.h5")

# 12. Save training history
import pickle
with open('../model/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("Training history saved to ../model/training_history.pkl")