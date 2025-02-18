import streamlit as st
from pathlib import Path
from utils import (
    load_model,  
    infer_uploaded_video, 
    infer_rtsp_stream,  
    process_and_save_video,
    process_and_save_images
)
import config

def main():
    st.title("Interactive Interface for YOLOv8")
    st.sidebar.header("DL Model Config")

    # Task Selection
    task_type = st.sidebar.selectbox("Select Task", ["Detection"])
    model_type = st.sidebar.selectbox(
        "Select Model", config.DETECTION_MODEL_LIST
    )
    confidence = st.sidebar.slider(
        "Confidence Threshold", min_value=30, max_value=100, value=50
    ) / 100.0

    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))

    # Load YOLOv8 Model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model. Error: {e}")

    # Source selection
    st.sidebar.header("Image/Video Config")
    source_selectbox = st.sidebar.selectbox(
        "Select Source", config.SOURCES_LIST
    )

    if source_selectbox == config.SOURCES_LIST[0]:  # Image
        infer_uploaded_video(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[1]:  # Video
        infer_rtsp_stream(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[2]:  # Webcam
        process_and_save_video(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[3]:  # RTSP
        process_and_save_images(confidence, model)
    else:
        st.error("Invalid source selection.")

    # Back to Main Menu
    if st.sidebar.button("Back to Main Menu"):
        st.session_state["current_page"] = "main"  # Switch back to main page
        st.rerun()


if __name__ == "__main__":
    main()
