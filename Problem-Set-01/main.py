# Import the necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Preprocess the data
train_ds = keras.preprocessing.image_dataset_from_directory(
    "/home/maruf/Documents/ml-dataset/Archive/train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(224, 224),
    batch_size=32,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    "/home/maruf/Documents/ml-dataset/Archive/val",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(224, 224),
    batch_size=32,
)
test_ds = keras.preprocessing.image_dataset_from_directory(
    "/home/maruf/Documents/ml-dataset/Archive/test",
    seed=1337,
    image_size=(224, 224),
    batch_size=32,
)

# Build the CNN model
model = keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Train the CNN model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Evaluate the CNN model
model.evaluate(test_ds)
