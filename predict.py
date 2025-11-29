import tensorflow as tf
import numpy as np
import os
import sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix



model = tf.keras.models.load_model("model/garbage_classifier_cnn_final.h5")

labels = ['battery','biological','brown-glass','cardboard',
          'clothes','green-glass','metal','paper',
          'plastic','shoes','trash','white-glass']

if len(sys.argv) > 1:
    img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print(f"\nImage not found: {img_path}")
        exit()

    img = load_img(img_path, target_size=(128,128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    print("\n Predicted Class:", labels[np.argmax(pred)].upper())
    exit()




TEST_DIR = "dataset/content/garbage_split/test"   

datagen = ImageDataGenerator(rescale=1/255)
test_data = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(128,128),
    batch_size=32,
    class_mode="sparse",
    shuffle=False
)


loss, acc = model.evaluate(test_data)
print(f"\n TEST ACCURACY: {round(acc*100,2)}%")
print(f" TEST LOSS: {round(loss,4)}")


y_pred = np.argmax(model.predict(test_data), axis=1)
y_true = test_data.classes

print("\n CLASSIFICATION REPORT:\n")
print(classification_report(y_true, y_pred, target_names=labels))

print("\n CONFUSION MATRIX:\n")
print(confusion_matrix(y_true, y_pred))

print("\n✔ Done.")
