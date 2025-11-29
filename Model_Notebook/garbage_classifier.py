


from google.colab import drive
drive.mount('/content/drive')

# !unzip "/content/drive/MyDrive/Garbage-Classifier/archive.zip" -d "/content/garbage_dataset"

# !pip install split-folders

import splitfolders
splitfolders.ratio(
    "/content/garbage_dataset/garbage_classification",
    output="/content/garbage_split",
    seed=42,
    ratio=(0.7, 0.2, 0.1)
)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1/255)

train = datagen.flow_from_directory("/content/garbage_split/train",
                                    target_size=(128,128),
                                    batch_size=32,
                                    class_mode="sparse")

val = datagen.flow_from_directory("/content/garbage_split/val",
                                  target_size=(128,128),
                                  batch_size=32,
                                  class_mode="sparse")

test = datagen.flow_from_directory("/content/garbage_split/test",
                                   target_size=(128,128),
                                   batch_size=32,
                                   class_mode="sparse",
                                   shuffle=False)

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(12,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(train, validation_data=val, epochs=15)

test_loss, test_acc = model.evaluate(test)
print("\ REAL TEST ACCURACY:", round(test_acc*100,2), "%")

model.save("/content/garbage_classifier_cnn_final.h5")
print("Model saved successfully!")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

pred = model.predict(test)
y_pred = np.argmax(pred, axis=1)
y_true = test.classes
labels = list(test.class_indices.keys())

print("\CLASSIFICATION REPORT:\n")
print(classification_report(y_true, y_pred, target_names=labels))

print("\ CONFUSION MATRIX:\n")
print(confusion_matrix(y_true, y_pred))

