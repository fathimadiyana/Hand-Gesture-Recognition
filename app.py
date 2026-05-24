import os
import zipfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

st.title("✋ Hand Gesture Recognition")
st.write("SkillCraft Technology - Machine Learning Task 4")
st.write("This app trains a model to classify hand gestures from image data.")

ZIP_PATH = "archive.zip"
DATASET_PATH = "leapGestRecog"
MODEL_PATH = "hand_gesture_model.pkl"

# Extract dataset
if not os.path.exists(DATASET_PATH):
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")
        st.success("Dataset extracted successfully!")
    else:
        st.error("archive.zip not found. Please keep archive.zip in the same folder as app.py.")
        st.stop()

# Load images
images = []
labels = []

st.info("Loading gesture images... Please wait.")

for subject in os.listdir(DATASET_PATH):
    subject_path = os.path.join(DATASET_PATH, subject)

    if os.path.isdir(subject_path):
        for gesture in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture)

            if os.path.isdir(gesture_path):
                count = 0

                for img_name in os.listdir(gesture_path):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(gesture_path, img_name)

                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                        if img is not None:
                            img = cv2.resize(img, (64, 64))
                            images.append(img.flatten())
                            labels.append(gesture)
                            count += 1

                        # small limit to make training faster
                        if count >= 40:
                            break

X = np.array(images)
y = np.array(labels)

if len(X) == 0:
    st.error("No images found. Please check your dataset folder.")
    st.stop()

st.subheader("📌 Dataset Information")
st.write("Total images loaded:", len(X))
st.write("Gesture classes:", sorted(list(set(y))))

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

st.subheader("🧠 Train Model")

if st.button("Train Model"):
    with st.spinner("Training model... Please wait"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump((model, encoder), f)

        st.success("Model trained successfully!")
        st.write("Accuracy:", round(accuracy * 100, 2), "%")
        st.success("Model saved as hand_gesture_model.pkl")

st.subheader("📤 Upload Image for Prediction")

uploaded_file = st.file_uploader("Upload hand gesture image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    if not os.path.exists(MODEL_PATH):
        st.warning("Please train the model first before prediction.")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model, encoder = pickle.load(f)

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    img = np.array(image.convert("L"))
    img = cv2.resize(img, (64, 64))
    img = img.flatten().reshape(1, -1)

    prediction = model.predict(img)[0]
    probability = model.predict_proba(img)[0]

    gesture = encoder.inverse_transform([prediction])[0]
    confidence = max(probability) * 100

    st.success(f"Predicted Gesture: {gesture}")
    st.info(f"Confidence Score: {confidence:.2f}%")