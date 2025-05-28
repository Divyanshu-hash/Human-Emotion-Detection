# 🧠 Emotion Detector with Face Overlay

This project is a **Streamlit-based web application** that uses **DeepFace** to detect human emotions from an uploaded image. It overlays a bounding box on the detected face and displays the dominant emotion with a confidence breakdown.

---

## 🔍 Features

- ✅ Upload an image and detect **facial emotions** using DeepFace.
- ✅ Bounding box overlay on the detected face.
- ✅ Display **dominant emotion**.
- ✅ Show a **bar chart** of emotion confidence levels.
- ✅ Built with **Streamlit** for simplicity and speed.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV](https://opencv.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)
- Python 🐍

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/Divyanshu-hash/Human-Emotion-Detection.git
cd emotion-detector-face-overlay

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Run streamlit app
streamlit run main.py

