import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.set_page_config(page_title="Emotion Detector with Face Overlay", page_icon="🧠")

st.title("🧠 Emotion Detector Tool with Face Overlay")
st.markdown("Upload a human face image and see detected emotions with bounding boxes 📦")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')  # ensure RGB
    image_np = np.array(image)

    # Run DeepFace analysis
    with st.spinner("Analyzing... 🔍"):
        result = DeepFace.analyze(
            img_path=image_np,
            actions=['emotion'],
            detector_backend='retinaface',
            enforce_detection=False
        )

    if result and isinstance(result, list) and len(result) > 0:
        # Extract data
        face_data = result[0]
        emotion_scores = face_data['emotion']
        dominant_emotion = face_data['dominant_emotion']
        region = face_data['region']  # x, y, w, h

        # Draw bounding box
        image_with_box = cv2.cvtColor(image_np.copy(), cv2.COLOR_RGB2BGR)
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_with_box, dominant_emotion.upper(), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        image_with_box = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)

        # Display image
        st.image(image_with_box, caption="Detected Face with Emotion", use_container_width=True)

        # Emotion Distribution Chart
        st.markdown("## 📊 Emotion Distribution")
        fig, ax = plt.subplots()
        ax.bar(emotion_scores.keys(), emotion_scores.values(), color='skyblue')
        ax.set_ylabel("Confidence (%)")
        ax.set_xlabel("Emotions")
        ax.set_title("Emotion Probabilities")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)  # Prevent memory leaks

        # Summary
        st.success(f"🧠 Dominant Emotion: **{dominant_emotion.upper()}**")
        st.markdown("### 🔬 All Emotions:")
        for emotion, score in emotion_scores.items():
            st.write(f"- **{emotion.capitalize()}**: {score:.2f}%")
    else:
        st.warning("⚠️ No face detected in the image.")
else:
    st.info("📌 Please upload an image to get started.")
