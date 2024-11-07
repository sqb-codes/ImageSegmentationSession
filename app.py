# pip install streamlit --user
import streamlit as st
import test_model
from PIL import Image

st.set_page_config(layout="wide")
st.title("Nuclie Segementation App")

img = st.sidebar.selectbox(
    "Select Image", ("test_img_1.png",
                     "test_img_2.png",
                     "test_img_3.png",
                     "test_img_4.png",
                     "test_img_5.png",
                     "test_img_6.jpg",
                     "test_1.jpg","img_1.jpg","img_3.jpeg")
)

input_image = "test_images/" + img

image = Image.open(input_image)
st.image(image, width=400)

detect_mask = st.button("Detect Mask")

if detect_mask:
    mask = test_model.do_pred(input_image)
    st.write("Predicted Mask")
    st.image(mask, width=500)
