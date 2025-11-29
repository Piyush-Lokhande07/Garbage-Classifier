

# 🗑️♻ Garbage Classification using CNN (12-Class Image Recognition)

This project is a **Deep Learning based Garbage Classification System** built using a Convolutional Neural Network (CNN).
It can predict waste type across **12 categories** and can be used for **smart waste management, recycling automation, robotics & sustainability projects.**

---

## 🚀 Features

| Feature                                     | Status |
| ------------------------------------------- | :----: |
| 12-Class Garbage Image Classifier           |    ✔   |
| Custom CNN developed & trained from scratch |    ✔   |
| ~76% Test Accuracy                          |    ✔   |
| Model evaluatable anytime with test set     |    ✔   |
| Predict any image locally (CLI)             |    ✔   |
| Interactive Web UI using Streamlit          |    ✔   |
| Model stored using Git LFS                  |    ✔   |

---

## 📂 Dataset

Dataset contains **12 garbage categories**, each stored in separate folders:

```
battery, biological, brown-glass, cardboard,
clothes, green-glass, metal, paper,
plastic, shoes, trash, white-glass
```

Split used during training:

| Dataset    | Split |
| ---------- | ----- |
| Train      | 70%   |
| Validation | 20%   |
| Test       | 10%   |

Dataset is kept **locally** and not uploaded to GitHub (ignored via `.gitignore`)

---

## 🧠 Model Architecture (CNN)

| Layer Type       | Details              |
| ---------------- | -------------------- |
| Conv2D + MaxPool | 32 filters           |
| Conv2D + MaxPool | 64 filters           |
| Conv2D + MaxPool | 128 filters          |
| Flatten          | —                    |
| Dense Layer      | 256 neurons (ReLU)   |
| Dropout          | 0.3                  |
| Output Layer     | 12 neurons (softmax) |

Training Details:

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Loss function | sparse_categorical_crossentropy |
| Optimizer     | Adam                            |
| Epochs        | 15                              |

---

## 📊 Results

| Metric              | Value                |
| ------------------- | -------------------- |
| Final Test Accuracy | ~76.11%              |
| Validation Accuracy | Near similar         |
| Precision/Recall/F1 | Generated via script |

Evaluate anytime using:

```
python predict.py
```

You will get:

✔ Test Accuracy
✔ Full Classification Report
✔ Confusion Matrix

---

## 🔥 Local Prediction

For a single waste image:

```
python predict.py images/sample.jpg
```

Output Example:

```
🟢 Predicted Class: PLASTIC
```

For complete model evaluation:

```
python predict.py
```

---

## 🌐 Streamlit UI (Frontend)

Interactive Interface: `app.py`

Run locally:

```
streamlit run app.py
```

Upload an image → The classifier predicts instantly.

Deployment ready on Streamlit Cloud.

---

## 📁 Project Directory

```
Garbage-Classifier/
│
├── app.py                      ← Streamlit UI
├── predict.py                  ← CLI prediction + test report
├── model/
│   └── garbage_classifier_cnn_final.h5
├── dataset/ (local only - ignored)
│   ├── train/
│   ├── val/
│   └── test/
└── README.md
```

---

## 🧰 Tech Stack Used

| Technology             | Role                               |
| ---------------------- | ---------------------------------- |
| **Python**             | Core programming                   |
| **TensorFlow / Keras** | CNN model development & training   |
| **NumPy**              | Numerical processing               |
| **Pillow**             | Image loading & preprocessing      |
| **Scikit-Learn**       | Classification report & evaluation |
| **Streamlit**          | Web app UI                         |
| **GitHub + Git LFS**   | Version control & model hosting    |



