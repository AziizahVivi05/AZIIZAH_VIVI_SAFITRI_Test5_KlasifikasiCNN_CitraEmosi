import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# 1. Konfigurasi Parameter
IMG_WIDTH, IMG_HEIGHT = 48, 48
BATCH_SIZE = 64
EPOCHS = 25
NUM_CLASSES = 7  # angry, disgust, fear, happy, neutral, sad, surprise

# Direktori dataset (sesuaikan dengan path lokal setelah diunduh dari Kaggle)
# https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset/data
train_dir = r"D:\KlasifikasiCittraCNN\dataset\train"
test_dir = r"D:\KlasifikasiCittraCNN\dataset\test"

# 2. Pra-pemrosesan dan Data Augmentation (Langkah 3)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 # Membagi data train menjadi train & validation jika diperlukan
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 3. Pembangunan Arsitektur Model CNN (Langkah 4)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')  # Softmax untuk output layer
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Pelatihan Model (Training)[cite: 2]
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 5. Simpan Model[cite: 2]
model.save('emotion_cnn_model.h5')
print("Model berhasil disimpan sebagai 'emotion_cnn_model.h5'")

# 6. Evaluasi Model (Langkah 5)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# Laporan Klasifikasi (Accuracy, Precision, Recall, F1-Score)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))