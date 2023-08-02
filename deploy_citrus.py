from io import BytesIO
import streamlit as st
import subprocess
subprocess.check_call(['pip', 'install', 'tensorflow'])

# pip install tensorflow
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

st.set_option('deprecation.showfileUploaderEncoding', True)
# st.text("Provide URL of Flower image to be classified:")
@st.cache(allow_output_mutation=True)
def load_model1():
	model1 = tf.keras.models.load_model('models/citrus_leaves_test_88.h5')
	return model1
with st.spinner('Loading Model into Memory....'):
    model1 = load_model1()

class_names = ['Black spot', 'Melanose', 'Canker', 'Greening', 'healthy']  


def predict_class1(image, model1):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [224, 224])

	image = np.expand_dims(image, axis = 0)

	prediction1 = model1.predict(image)

	return prediction1
def predict_class2(image, model2):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [224, 224])

	image = np.expand_dims(image, axis = 0)

	prediction2 = model2.predict(image)

	return prediction2



model1 = load_model1()

st.image('Lemon.jpg', width=200, )
st.title('Citrus Plant Diseases Detection')

file = st.file_uploader("Upload an image of disease effected leaf", type=["jpg", "png", "jpeg"])

if file is None:
    st.text('Waiting for upload...')
    
else:
    slot = st.empty()
    slot.text('Running the Inference...')
    test_image = Image.open(file)
    st.image(test_image, caption='Input Image', width= 500)
    pred = predict_class1(np.asarray(test_image), model1)
    score = tf.nn.softmax(pred[0])
    output = "This image most likely belongs to {}".format(class_names[np.argmax(score)])#, 100 * np.max(score))
    st.success(output)
    

@st.cache(allow_output_mutation=True)
def load_model2():
	model2 = tf.keras.models.load_model('models/citrus_Fruits_default_46.hdf5')
	return model2
with st.spinner('Loading Model into Memory....'):
    model2 = load_model2()
class_names2 = ['Black spot', 'Canker', 'Greening', 'Scab', 'healthy']
   
file = st.file_uploader("Upload an image of disease effected fruit", type=["jpg", "png", "jpeg"])

if file is None:
    st.text('Waiting for upload...')
    
else:
    slot = st.empty()
    slot.text('Running the Inference...')
    test_image = Image.open(file)
    st.image(test_image, caption='Input Image', width= 500)
    pred = predict_class2(np.asarray(test_image), model2)
    score = tf.nn.softmax(pred[0])
    # output = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names2[np.argmax(score)])#, 100 * np.max(score))
    output = "This image most likely belongs to {}".format(class_names2[np.argmax(score)])
    st.success(output)
