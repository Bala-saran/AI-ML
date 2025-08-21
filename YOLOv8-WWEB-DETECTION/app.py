# For webcam and browser interface
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# For handling video frames
import av

# For YOLOv8 model
from ultralytics import YOLO

# Set the title and page layout
st.set_page_config(page_title="YOLOv8 Live", layout="wide")

st.title("YOLOv8 Live Object Detection")
st.write("Runs entirely in your browser. Press **Stop** to end.")
@st.cache_resource
def load_model(path: str):
    """Loads the YOLOv8 model only once."""
    return YOLO(path)

# Create three side-by-side selectors
col1, col2, col3 = st.columns(3)

# Model choice
model_name = col1.selectbox("Model", [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
], index=0)

# Confidence slider
conf = col2.slider("Confidence", 0.05, 0.90, 0.25, 0.01)

# Device choice (auto/CPU/GPU)
device = col3.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

# Optional: limit detection to specific classes
classes_text = st.text_input("Limit to classes (comma-separated, optional)")

# Load the chosen model
model = load_model(model_name)

# Convert class names to IDs
if classes_text.strip():
    name_to_id = {name: idx for idx, name in model.names.items()}
    selected = [s.strip() for s in classes_text.split(",") if s.strip() in name_to_id]
    CLASS_NAME_TO_ID = [name_to_id[s] for s in selected] if selected else None
else:
    CLASS_NAME_TO_ID = None

# WebRTC configuration for webcam streaming
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Convert frame to a NumPy array (OpenCV format)
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO detection
        results = model.predict(
            img,
            conf=conf,
            verbose=False,
            device=(None if device == "auto" else device),
            classes=CLASS_NAME_TO_ID
        )

        # Draw bounding boxes on the frame
        annotated = results[0].plot()

        # Send the annotated frame back
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="yolov8-live",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration=rtc_config,
)


