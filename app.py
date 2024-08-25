import streamlit as st
import cv2
from cv2 import dnn_superres
import numpy as np
from PIL import Image, ImageEnhance

# Function to handle the image upscaling
def upscale_image(image, scale=2):
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Load the desired model
    model_path = "FSRCNN_x2.pb"  # Make sure you have the correct path to your model
    sr.readModel(model_path)

    # Set CUDA backend and target to enable GPU inference (if supported)
    # sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Set the desired model and scale
    sr.setModel("fsrcnn", scale)

    # Upscale the image
    result = sr.upsample(image)

    return result

# Set up the page configuration
st.set_page_config(page_title="Image Upscaler", page_icon="🔍", layout="wide")

# Custom CSS for dark gradient background and styling
st.markdown("""
    <style>
    /* Apply a dark gradient background to the entire page */
    .main {
        background: linear-gradient(135deg, #1c1c1c, #444444);
        color: #ffffff;
    }

    /* Style for the title */
    h1 {
        background-color: #1c1c1c; /* Dark background */
        color: #ff6666;             /* Subtle red font */
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }

    /* Style for buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }

    /* Style for file uploader */
    .stFileUploader {
        font-size: 16px;
        font-weight: bold;
    }

    /* Style for images */
    .stImage {
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.4);
    }

    /* Style the sidebar */
    .sidebar .sidebar-content {
        background: #2c2c2c;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for additional settings
st.sidebar.title("🔧 Image Settings")
brightness = st.sidebar.slider("Brightness", 0.5, 3.0, 1.0)
contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
sharpness = st.sidebar.slider("Sharpness", 0.5, 3.0, 1.0)
saturation = st.sidebar.slider("Saturation", 0.5, 3.0, 1.0)
hue = st.sidebar.slider("Hue", -0.5, 0.5, 0.0)

# Main page title
st.title("🖼️ Image Upscaler using Super Resolution")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image to upscale...", type=["bmp", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    input_image_cv = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)                 # Converting image into a compatible format for OpenCV
    
    # Display the input image and its details
    st.image(input_image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True, clamp=True)
    st.write(f"**Image Resolution:** {input_image.size[0]} x {input_image.size[1]}")
    
    # Process the image with the super resolution model
    output_image_cv = upscale_image(input_image_cv)
    
    output_image = Image.fromarray(cv2.cvtColor(output_image_cv, cv2.COLOR_BGR2RGB))        # Convert the output image back to PIL format
    
    # Apply brightness and contrast adjustments
    enhancer = ImageEnhance.Brightness(output_image)
    output_image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(output_image)
    output_image = enhancer.enhance(contrast)
    
    # Apply sharpness adjustment
    enhancer = ImageEnhance.Sharpness(output_image)
    output_image = enhancer.enhance(sharpness)

    # Apply saturation adjustment
    enhancer = ImageEnhance.Color(output_image)
    output_image = enhancer.enhance(saturation)

    # Apply hue adjustment (convert to HSV, adjust hue, and convert back)
    if hue != 0.0:
        output_image = output_image.convert("HSV")
        np_image = np.array(output_image)
        np_image[..., 0] = np.mod(np_image[..., 0].astype(int) + int(hue * 255), 255)
        output_image = Image.fromarray(np_image, mode="HSV").convert("RGB")
    
    # Display the output image and its details
    st.subheader("Upscaled Image with Adjustments")
    st.image(output_image, caption="Upscaled Image", use_column_width=True, clamp=True)
    st.write(f"**Upscaled Image Resolution:** {output_image.size[0]} x {output_image.size[1]}")

    # Provide a download button for the output image and generate a dynamic filename based on adjustments
    output_filename = f"SuperResOutput_brightness_{brightness:.2f}_contrast_{contrast:.2f}_sharpness_{sharpness:.2f}_saturation_{saturation:.2f}_hue_{hue:.2f}.png"
    output_image_path = output_filename
    output_image.save(output_image_path)
    with open(output_image_path, "rb") as file:
        st.download_button(label="💾 Download Upscaled Image", data=file, file_name=output_filename, mime="image/png")
else:
    st.info("Please upload an image file to upscale.")