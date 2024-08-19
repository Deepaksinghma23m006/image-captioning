import streamlit as st
import pickle
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to load pickle files
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Load saved files
tokenizer = load_pickle('tokenizer.pkl')
features = load_pickle('features.pkl')
caption_model = load_model('caption_model.h5')

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, feature, tokenizer, max_length):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text

# Streamlit App
st.title("Image Captioning App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Save uploaded image to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    img = load_img(tmp_file_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Assuming a feature extraction model is defined and loaded as `fe`
    # Example: fe = load_model("path_to_your_feature_extraction_model.h5")
    feature = fe.predict(img, verbose=0)

    # Predict caption
    caption = predict_caption(caption_model, feature, tokenizer, max_length=34)
    st.write("Caption:", caption)

    # Clean up temporary file
    os.remove(tmp_file_path)
