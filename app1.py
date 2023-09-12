import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
# import io


# LINK TO THE CSS FILE
with open(r"C:\Users\Rishitha Reddy\OneDrive\Desktop\Brain_Tumor\style.css")as f:
 st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("images/bg.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# add_bg_from_url() 

st.write("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

# @st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model(r"C:\Users\Rishitha Reddy\OneDrive\Desktop\Brain_Tumor\models\model_cnn.h5")
  return model

with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Brain Tumor Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file here", type=["jpg", "png","hpeg"])

st.set_option('deprecation.showfileUploaderEncoding', False)



if file is None:
    st.text("Please upload an image file")
else:
    image1 = Image.open(file)
    image = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image,(256,256))
    image = tf.reshape(image, [1, 256, 256,3])
    prediction = model.predict(image)
    st.image(image1, use_column_width=True)
    score = tf.nn.softmax(prediction[0])
    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
    # st.write(np.argmax(score))
    if class_names[np.argmax(score)]=="Glioma Tumor":
        st.write('<p class="big-font">Glioma Tumor</p>', unsafe_allow_html=True)
    elif class_names[np.argmax(score)]=="Meningioma Tumor":
        st.write('<p class="big-font">Meningioma Tumor</p>', unsafe_allow_html=True)
    elif class_names[np.argmax(score)]=="No Tumor":
        st.write('<p class="big-font">No Tumor</p>', unsafe_allow_html=True)
    else:
       st.write('<p class="big-font">Pituitary Tumor</p>', unsafe_allow_html=True)
