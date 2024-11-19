import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import tempfile
from streamlit_option_menu import option_menu  # Install with 'pip install streamlit-option-menu'

# Set parameters
image_size_xray = (224, 224)  # X-ray image size
image_size_ct = (350, 350)    # CT scan image size



@st.cache_resource
def load_pneumonia_model():
    # URL to the pre-trained pneumonia detection model
    url = "https://pnemonia.s3.us-east-1.amazonaws.com/pneumonia_detection_model.h5"

    # Check if model is already cached, if not, download it
    try:
        # Check if the model file exists in the local temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            model_path = temp_file.name

            # Download model if not cached
            st.info("Downloading model...")
            response = requests.get(url)
            response.raise_for_status()

            # Write model content to temporary file
            with open(model_path, 'wb') as model_file:
                model_file.write(response.content)

            st.success("Model loaded successfully.")
            return load_model(model_path)

    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")






@st.cache_resource
def load_edema_model():
    return tf.keras.models.load_model(os.path.join(os.getcwd(), 'edema_detection_model.h5'))

@st.cache_resource
def load_cancer_model():
    return tf.keras.models.load_model(os.path.join(os.getcwd(), 'trained_lung_cancer_model.h5'))

# Preprocess the input image
def load_and_preprocess_image(uploaded_image, image_size, is_xray=False):
    uploaded_image = np.array(uploaded_image)  # Convert the image to a NumPy array
    
    if is_xray:
        # If it's an X-ray image (grayscale), convert it to RGB by repeating across the third dimension
        if uploaded_image.ndim == 2:  # Grayscale image (height, width)
            uploaded_image = np.stack([uploaded_image] * 3, axis=-1)  # Convert to (height, width, 3)
    
    uploaded_image = tf.convert_to_tensor(uploaded_image)  # Convert to a Tensor
    uploaded_image = tf.image.resize(uploaded_image, image_size) / 255.0  # Resize and normalize
    uploaded_image = tf.expand_dims(uploaded_image, axis=0)  # Add batch dimension
    return uploaded_image

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Abnormality Detection",  # Menu title
        options=["Home", "Pneumonia Detection", "Edema Detection", "Lung Cancer Detection"],  # Pages
        icons=["house", "lungs", "lungs", "lungs"],  # Icons for each page
        menu_icon="cast",  # Menu icon
        default_index=0,  # Default selected page
    )

# Home Page
if selected == "Home":
    st.title("Abnormality Detection System")
    st.write("Welcome to the X-ray and CT scan analysis tool. Use the sidebar to navigate between detection options.")
    st.image("https://i.pinimg.com/originals/4f/4d/c5/4f4dc52c73af1d9721c7ec1410e880ff.jpg", use_column_width=True)
    st.markdown("""
        **Features:**
        - Detect Pneumonia from X-ray images.
        - Detect Edema from X-ray images.
        - Detect Cancer from CT scan images.
        - Detailed insights into predictions.
        - Easy-to-use interface for doctors and researchers.
    """)

# Pneumonia Detection Page
elif selected == "Pneumonia Detection":
    st.title("Pneumonia Detection")
    st.write("Upload an X-ray image to check for Pneumonia.")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    print("Everythin okay")
    print(type(uploaded_file))
    
    if uploaded_file is not None:
        print(type(uploaded_file))
        uploaded_image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        st.image(uploaded_image, caption="Uploaded X-ray", use_column_width=True)
        
        with st.spinner("Analyzing the image..."):
            img_array = load_and_preprocess_image(uploaded_image, image_size_xray, is_xray=True)
            prediction = load_pneumonia_model().predict(img_array)[0][0]
            print("Hii")

        progress = int(prediction * 100)
        st.progress(progress)

        if prediction > 0.7:
            st.warning("**Prediction:** Pneumonia detected with high confidence.")
            #st.markdown(f"Confidence Level: **{prediction * 100:.2f}%**")
        elif prediction < 0.3:
            st.success("**Prediction:** No Pneumonia detected.")
            #st.markdown(f"Confidence Level: **{(1 - prediction) * 100:.2f}%**")
        else:
            st.info("**Prediction:** Uncertain, further inspection advised.")
            #st.markdown(f"Confidence Level: **{prediction * 100:.2f}%**")

# Edema Detection Page
elif selected == "Edema Detection":
    st.title("Edema Detection")
    st.write("Upload an X-ray image to check for Edema.")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        st.image(uploaded_image, caption="Uploaded X-ray", use_column_width=True)
        
        with st.spinner("Analyzing the image..."):
            img_array = load_and_preprocess_image(uploaded_image, image_size_xray, is_xray=True)
            prediction = load_edema_model().predict(img_array)[0][0]

        progress = int(prediction * 100)
        st.progress(progress)

        if prediction > 0.60:
            st.warning("**Prediction:** Edema detected with high confidence.")
            #st.markdown(f"Confidence Level: **{prediction * 100:.2f}%**")
        else:
            st.success("**Prediction:** No Edema detected.")
            #st.markdown(f"Confidence Level: **{(1 - prediction) * 100:.2f}%**")

# Cancer Detection Page
elif selected == "Lung Cancer Detection":
    st.title("Cancer Detection")
    st.write("Upload a CT scan image to check for Cancer.")

    uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")  # CT scan is expected in RGB
        st.image(uploaded_image, caption="Uploaded CT scan", use_column_width=True)
        
        with st.spinner("Analyzing the image..."):
            img_array = load_and_preprocess_image(uploaded_image, image_size_ct, is_xray=False)
            prediction = load_cancer_model().predict(img_array)[0][0]

        progress = int(prediction * 100)
        st.progress(progress)

        if prediction < 0.3:
            st.warning("**Prediction:** Cancer detected with high confidence.")
            #st.markdown(f"Confidence Level: **{prediction * 100:.2f}%**")
        elif prediction > 0.7:
            st.success("**Prediction:** No Cancer detected.")
            #st.markdown(f"Confidence Level: **{(1 - prediction) * 100:.2f}%**")
        else:
            st.info("**Prediction:** Uncertain, further inspection advised.")
            #st.markdown(f"Confidence Level: **{prediction * 100:.2f}%**")
