#
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import os  # Make sure os is imported
#
# # Debug: Check current working directory and files
# # st.write("📁 Current working directory:", os.getcwd())
# # st.write("📄 Files in current directory:", os.listdir())
#
#
# #Tensorflow Model Prediction
# def model_prediction(test_image):
#     # model = tf.keras.models.load_model("trained_plant_disease_model.keras")
#     # model = tf.keras.models.load_model("trained_plant_disease_model.keras", compile=False)
#     # Try HDF5 format model
#     model = tf.keras.models.load_model("trained_model.h5", compile=False)
#
#     image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr]) #convert single image to batch
#     predictions = model.predict(input_arr)
#     return np.argmax(predictions) #return index of max element
#
# #Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])
#
# #Main Page
# if(app_mode=="Home"):
#     st.header("PLANT DISEASE RECOGNITION SYSTEM")
#     image_path = "home_page.jpeg"
#     st.image(image_path, use_container_width=True)
#
#     st.markdown("""
#     Welcome to the Plant Disease Recognition System!🌿
#
#     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
#
#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.
#
#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
#
#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
#
#     ### About Us
#     Learn more about the project, our team, and our goals on the **About** page.
#     """)
#
# #About Project
# elif(app_mode=="About"):
#     st.header("About")
#     st.markdown("""
#                 #### About Dataset
#                 This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
#                 This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
#                 A new directory containing 33 test images is created later for prediction purpose.
#                 #### Content
#                 1. train (70295 images)
#                 2. test (33 images)
#                 3. validation (17572 images)
#
#                 """)
#
# #Prediction Page
# elif(app_mode=="Disease Recognition"):
#     st.header("Disease Recognition")
#     test_image = st.file_uploader("Choose an Image:")
#     if(st.button("Show Image")):
#         st.image(test_image, width=4, use_container_width=True)
#
#     #Predict button
#     if(st.button("Predict")):
#         st.snow()
#         st.write("Our Prediction")
#         result_index = model_prediction(test_image)
#         #Reading Labels
#         class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#                     'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
#                     'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#                     'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
#                     'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#                     'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#                     'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
#                     'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
#                     'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
#                     'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
#                     'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
#                     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
#                     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#                       'Tomato___healthy']
#         st.success("Model is Predicting it's a {}".format(class_name[result_index]))


import streamlit as st
import tensorflow as tf
import numpy as np
import os

st.set_page_config(page_title="Plant Disease Recognition", page_icon="🌿", layout="centered")


# Model prediction function
def model_prediction(test_image):
    # model = tf.keras.models.load_model("trained_model.h5", compile=False)
    model = tf.keras.models.load_model("mobilenet_plant_disease.keras", compile=False)

    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# Sidebar
with st.sidebar:
    st.title("🌿 Plant Disease Dashboard")
    app_mode = st.selectbox("🔍 Navigate", ["Home", "About", "Disease Recognition"])
    st.markdown("---")
    st.caption("Developed by Tanya")
    st.markdown("[GitHub](https://github.com/) • [Contact](mailto:you@example.com)")

# Home
if app_mode == "Home":
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("""
    <h2 style='text-align:center;'>🌾 Plant Disease Recognition System</h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    Welcome to the **Plant Disease Recognition System**!  
    Upload an image of a plant leaf and our AI will detect potential diseases.

    ### 🚀 Features
    - **🔬 High Accuracy:** Trained on 87K+ images
    - **⚡ Fast Results:** Get predictions in seconds
    - **📱 Easy Interface:** No coding required

    ### 🧭 How It Works
    1. 📷 Upload an image on the **Disease Recognition** tab
    2. 🧠 Let the model analyze it
    3. ✅ Get the diagnosis instantly

    """, unsafe_allow_html=True)

# About
elif app_mode == "About":
    st.markdown("## 📚 About the Project")
    st.markdown("""
    This project uses a deep learning model trained on a large dataset of plant leaf images to classify and detect diseases.

    **Dataset Source:** Kaggle  
    **Classes:** 38 (including healthy and diseased leaves)  
    **Size:** ~87,000 images  
    **Split:** 80% training | 20% validation | 33 test images

    The model uses a **Convolutional Neural Network (CNN)** architecture with real-time image classification integrated via **Streamlit**.
    """)

# Disease Recognition
elif app_mode == "Disease Recognition":
    st.markdown("## 🦠 Disease Recognition")
    st.markdown("Upload an image of a plant leaf to identify the disease.")

    test_image = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict"):
            st.snow()
            result_index = model_prediction(test_image)
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            st.success(f"🌱 **Prediction:** {class_name[result_index]}")







