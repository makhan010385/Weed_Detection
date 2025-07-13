import streamlit as st
import pandas as pd
import os
import zipfile
import tempfile
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🌿 Weed Detection using SVM, Naive Bayes and CNN")

# --- File Upload ---
st.header("1️⃣ Upload Image Dataset & Labels")
image_data = st.file_uploader("Upload a .zip of images", type=['zip'])
label_file = st.file_uploader("Upload labels.csv", type=['csv'])

if image_data and label_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = os.path.join(temp_dir, image_data.name)
        with open(archive_path, 'wb') as f:
            f.write(image_data.read())

        # Extract ZIP
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Load labels
        df_labels = pd.read_csv(label_file)

        # Gather all images
        all_images = []
        for root, dirs, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(root, f))

        image_files_map = {Path(f).name: f for f in all_images}
        st.success(f"✅ Found {len(all_images)} image files.")

        # --- Prepare Data ---
        X = []
        y = []
        for _, row in df_labels.iterrows():
            fname = row['image_filename']
            label = row['label']
            img_path = image_files_map.get(fname)
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))
                X.append(img)
                y.append(label)

        if len(X) == 0:
            st.error("No valid images found.")
            st.stop()

        X = np.array(X) / 255.0
        X_flat = X.reshape(len(X), -1)  # For SVM & NB
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # Class count
        n_classes = len(np.unique(y_enc)) if len(np.unique(y_enc)) > 1 else 2

        # Safe target_names extraction
        if hasattr(le, "classes_") and hasattr(le.classes_, "__iter__") and not isinstance(le.classes_, str):
            target_names = [str(c) for c in le.classes_]
        else:
            target_names = [str(le.classes_)]

        # Train/test split
        X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X, y_enc, test_size=0.2, random_state=42)
        X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(X_flat, y_enc, test_size=0.2, random_state=42)

        # --- Model Selection ---
        st.header("2️⃣ Select a Model to Train and Evaluate")
        model_choice = st.selectbox("Choose a model", ["SVM", "Naive Bayes", "CNN"])

        # --- SVM ---
        if model_choice == "SVM":
            st.subheader("🔷 SVM Model Results")
            svm_model = SVC(kernel='linear')
            svm_model.fit(X_train_flat, y_train_flat)
            y_pred = svm_model.predict(X_test_flat)
            st.code(classification_report(y_test_flat, y_pred, target_names=target_names))

            cm = confusion_matrix(y_test_flat, y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax)
            st.pyplot(fig)

        # --- Naive Bayes ---
        elif model_choice == "Naive Bayes":
            st.subheader("🔶 Naive Bayes Model Results")
            nb_model = GaussianNB()
            nb_model.fit(X_train_flat, y_train_flat)
            y_pred = nb_model.predict(X_test_flat)
            st.code(classification_report(y_test_flat, y_pred, target_names=target_names))

            cm = confusion_matrix(y_test_flat, y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax)
            st.pyplot(fig)

        # --- CNN ---
        elif model_choice == "CNN":
            st.subheader("🔷 CNN Model Results")
            y_train_cat = to_categorical(y_train_cnn, num_classes=n_classes)
            y_test_cat = to_categorical(y_test_cnn, num_classes=n_classes)

            cnn_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(n_classes, activation='softmax')
            ])

            cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            cnn_model.fit(X_train_cnn, y_train_cat, epochs=5, validation_data=(X_test_cnn, y_test_cat), verbose=1)

            y_pred_probs = cnn_model.predict(X_test_cnn)
            y_pred = np.argmax(y_pred_probs, axis=1)
            st.code(classification_report(y_test_cnn, y_pred, target_names=target_names))

            cm = confusion_matrix(y_test_cnn, y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax)
            st.pyplot(fig)

            # --- Prediction from Upload ---
            st.header("3️⃣ Try CNN Prediction")
            test_image = st.file_uploader("Upload an image to classify (CNN only)", type=['jpg', 'jpeg', 'png'])

            if test_image:
                file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                img_resized = cv2.resize(img, (64, 64)) / 255.0
                prediction = cnn_model.predict(np.expand_dims(img_resized, axis=0))
                pred_label = le.inverse_transform([np.argmax(prediction)])[0]
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"🔍 Predicted: {pred_label}")
