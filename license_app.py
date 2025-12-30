import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

st.title("🪖 License Plate Detection with YOLOv11")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("license_plate.pt")

model = load_model()

st.write("Upload image or video to process")

# --- Image Upload ---
img_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
if img_file:
    st.image(img_file, caption="Original Image", use_column_width=True)
    img_bytes = img_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)
    out_frame = results[0].plot()
    st.image(out_frame, caption="Processed Image", use_column_width=True)

# --- Video Upload ---
vid_file = st.file_uploader("Upload Video", type=["mp4","avi","mov","mkv"])
if vid_file:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(vid_file.read())

    # Show original
    st.write("### Original Video")
    st.video(tfile.name)

    cap = cv2.VideoCapture(tfile.name)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # Use VP8 codec and WebM container
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    out_path = "processed.webm"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    st.write("Processing video… this can take time")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = model(frame)
        frame_out = res[0].plot()
        out.write(frame_out)

    cap.release()
    out.release()

    # Display processed video
    st.success("✔ Processed Video Ready")
    with open(out_path, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)
