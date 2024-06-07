import streamlit as st
from PIL import Image

st.title("Klasifikacija rijetkih biljaka i životinja")

classification_method = st.selectbox(
    'Odaberite model za klasifikaciju',
    ('Swin_V2 - najbolji za klasifikaciju životinja', 'ResNet50 - najbolji za klasifikaciju biljaka', 'MyFSL')
)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if st.button('Process Image'):
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.',  width=image.width)
        
        st.text("Processing the image...")

        result = f""

        st.text(result)
    else:
        st.text("Please upload an image before clicking the button.")
