import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import gdown

word_to_index = imdb.get_word_index()

def sentiment_classification(new_review_text, model):
    max_review_length = 500
    new_review_tokens = [word_to_index.get(word, 0) for word in new_review_text.split()]
    new_review_tokens = pad_sequences([new_review_tokens], maxlen=max_review_length)
    prediction = model.predict(new_review_tokens)
    if type(prediction) == list:
        prediction = prediction[0]
    return "Positive" if prediction > 0.5 else "Negative"

def tumor_detection(img, model):
    img = Image.open(img)
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    return "Tumor Detected" if res else "No Tumor"

# Display the title
st.title("ML MODEL")

# Select the task (Image classification or Text processing)
option = st.selectbox("Why are you here?", ('Image classification', 'Text processing'), index=None,
                     placeholder="Select problem method...")

st.write('You selected:', option)

if option == 'Image classification':
    out = st.radio(
        "Select your prediction 👉",
        key="visibility",
        options=["Tumor prediction"],
    )
    
    if out == "Tumor prediction":
        st.subheader("Tumor Detection")
        uploaded_file = st.file_uploader("Choose a tumor image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the tumor detection model
        model_link = "https://drive.google.com/uc?id=1ebEd5VeAacGS20yiw1qteQboqoPltD_x"
        output_file = 'CN.keras'
        # Load the tumor detection model
        gdown.download(model_link, output_file, quiet=False)
        model = load_model(output_file)
        st.image(uploaded_file, caption="Image.", use_column_width=False, width=350)
        st.write("")

        if st.button("Predict"):
            result = tumor_detection(uploaded_file, model)
            st.subheader("Tumor Detection Result")
            st.write(f"**{result}**")       
    
else:
    out = st.radio("Select your prediction 👉", options=["IMDB sentiment"])
    if out == "IMDB sentiment":
        new_review_text = st.text_area("Enter a New Review:", value="")
        if st.button("Submit") and not new_review_text.strip():
            st.warning("Please enter a review.")
            
        if new_review_text.strip():
            tasks = ["Perceptron", "LSTM", "Back Propagation", "RNN", "DNN"]
            inputs = st.radio("Select", tasks, horizontal=True)
            if inputs == "Perceptron":
                with open('PP.pkl', 'rb') as file:
                    model = pickle.load(file)

            elif inputs == "Back Propagation":
                with open('BP.pkl', 'rb') as file:
                    model = pickle.load(file)
                
            elif inputs == "LSTM":
                model = load_model('LS.keras')
            elif inputs == "DNN":
                model = load_model('DP.keras')
            elif inputs == "RNN":
                model = load_model('RN.keras')
            if st.button("Predict"):
                result = sentiment_classification(new_review_text, model)
                st.subheader("Sentiment Classification Result")
                st.write(f"**{result}**")
