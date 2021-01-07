''' THIS IS THE CODE FOR RUNNING IN STREAMLIT '''
import numpy as np 
import os
import io
from typing import Callable, List, NamedTuple, Tuple
import datetime
from gtts import gTTS 
import webbrowser
from io import BytesIO

# Language in which you want to convert 
language = 'en'
     
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)

from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz")

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

def get_resources_markdown() -> str:
    """Some info regarding Resources
    Arguments:
        model {KerasApplication} -- The KerasApplication employed
    Returns:
        str -- A Markdown string with links to relevant resources
    """
    return f"""### Resources
- [Keras](https://keras.io/)
- [AllenNLP](https://demo.allennlp.org/reading-comprehension)
   
- Know more
  - [AlexNet](https://en.wikipedia.org/wiki/AlexNet)
  - [Monuments of India](https://www.holidify.com/collections/monuments-of-india)
  - [Monument Recognition Using Deep Neural Networks](https://www.researchgate.net/publication/328820153_Monument_Recognition_Using_Deep_Neural_Networks)
  - [Image based Indian monument recognition using convoluted neural networks](https://www.researchgate.net/publication/324493420_Image_based_Indian_monument_recognition_using_convoluted_neural_networks)
  - [Understanding of a Convolutional Neural Network](https://www.researchgate.net/publication/319253577_Understanding_of_a_Convolutional_Neural_Network)
"""

class KerasApplication():
    """We wrap a Keras Application into this class for ease of use"""
    name: str
    keras_application: Callable
    input_shape: Tuple[int, int] = (227, 227)
    url: str = "https://keras.io/applications/"

    def load_image(self, image_path: str) -> Image:
        """Loads the image from file
        Arguments:
            image_path {str} -- The absolute path to the image
        Returns:
            Image -- The image loaded
        """
        return load_img(image_path, target_size=self.input_shape)

    def to_input_shape(self, image: Image) -> Image:
        """Resizes the image to the input_shape
        Arguments:
            image {Image} -- The image to reshape
        Returns:
            Image -- The reshaped image
        """
        im = Image.open(image)
        return im.resize(self.input_shape)

    def preprocess_input(self, image: Image) -> Image:
        """Prepares the image for classification by the classifier
        Arguments:
            image {Image} -- The image to preprocess
        Returns:
            Image -- The preprocessed image
        """
        image = self.to_input_shape(image)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    def get_top_predictions(self, image: Image = None, report_progress_func=print) -> List[Tuple[str, str, float]]:
        """[summary]
        Keyword Arguments:
            image {Image} -- An image (default: {None})
            report_progress_func {Callable} -- A function like 'print', 'st.write' or similar
            (default: {print})
        Returns:
            [type] -- The top predictions as a list of 3-tuples on the form
            (id, prediction, probability)
        """
        report_progress_func(
            f"Loading model ... (The first time this is done it may take several "
            "minutes)",
            10,
        )
        model = load_model("AISUCCESS3_with_new_train.h5")

        report_progress_func(f"Processing image ... ", 67)
        image = self.preprocess_input(image)
        report_progress_func(f"Classifying image ... ", 85)
        classes = model.predict_classes(image, batch_size=10)
        spot={0:'Agra Fort',1:'Ajanta and Ellora Caves',2:'Amer Fort',3:'Bangalore Palace',4:'Basilica of Bom Jesus',5:'Bekal Fort',6:'Charminar',7:'City Palace',8:'Elephanta Cave',9:'Fatehpur Sikri',10:'Gateway of India',11:'Gingee Fort',12:'Golden Temple',13:'Golkonda Fort',14:'Gwalior Fort',15:'Hawa Mahal',16:'Hill Palace',17:'Howrah Bridge',18:'Humayuns Tomb',19:'India Gate',20:'Jama Masjid',21:'Janta Mantir',22:'Kaye Monastry',23:'Konark Sun Temple',24:'Lotus Temple',25:'Madurai Meenakshi Temple',26:'Mysore Palace',27:'Nalanda University',28:'Qutub Minar',29:'Ran ki Vav',30:'Rashtrapati Bhavan',31:'Red Fort',32:'Sanchi Stupa',33:'Shore Temple Mahabalipuram',34:'Taj Mahal',35:'Thanjavur Chola Temple',36:'Victoria Memorial',37:'Victoria Terminal',38:'Vidhana Soudha',39:'Vivekananda Rock Memorial'}
        place=spot[classes[0]]        
        report_progress_func("", 0)
        return place

DEFAULT_KERAS_APPLICATION_INDEX = 0
KERAS_APPLICATIONS: List[KerasApplication] = [
    KerasApplication(
    )
]      

def classify():
    st.image('https://t3.ftcdn.net/jpg/00/96/55/94/240_F_96559467_Fxgsa20HIuPGWywzEDnBMy3NokapCzxH.jpg',width=420, use_column_width=True)
    st.write("Hi! May I help you with the place?")
    st.sidebar.markdown(get_resources_markdown())
    st.sidebar.write("  \n")
    selected_model = st.sidebar.selectbox(
        ("Image Classifier Model"),
        options=KERAS_APPLICATIONS,
        index=DEFAULT_KERAS_APPLICATION_INDEX,
    )
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        progress_bar = st.empty()
        progress = st.empty()

        def report_progress(message, value, progress=progress, progress_bar=progress_bar):
            if value == 0:
                progress_bar.empty()
                progress.empty()
            else:
                progress_bar.progress(value)
                progress.markdown(message)
        predictions = selected_model.get_top_predictions(
            image=uploaded_file, report_progress_func=report_progress
        )
        st.write(" ")
        st.write("Classified")
        st.subheader('You are at %s ' % predictions)
    return predictions

def chat(passage):
    st.write("Excited to know more about the place?!")
    st.write("Here's HAL9000 to help you with your queries!")
    st.write("**type _bye_ to end the chat")
    while True:
        inp = st.text_input("You: ", key='1')
        
        if inp.lower() == "bye":
            st.text_area("Bot:", value="Thank You, Have a nice day!", height=200, max_chars=None, key=None)
            break
        result = predictor.predict(passage=passage,question=inp)
        response = result['best_span_str']
        st.text_area("Bot:", value=response, height=200, max_chars=None, key=None)
        
        mytext = response
        myobj = gTTS(text=mytext, lang=language, slow=False)
        date_string = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
        filename = "voice"+date_string+".ogg"
        myobj.save(filename)
        name = filename
        audio_file = open(name, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')
        os.remove(name)
        
def main():
    menu = ["Chatbot","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Chatbot":
        st.title("HAL9000")
        st.subheader(" Welcome to your visual chatbot assistant HAL9000 v.2.1.0.")
        location=classify()
        
        x=["Yes","No"]
        st.write(" Did I Get it Right?")
        choose=st.selectbox("Choose",x)
        if choose == "Yes":
            name = "Text_Data/"+location+".txt"
            with open(name, mode="r",encoding="utf-8" ) as input_file:
                passage = input_file.read()
            chat(passage)
        else:
            st.write(" I'm really sorry for that.")
            labels=['Agra Fort','Ajanta and Ellora Caves','Amer Fort','Bangalore Palace','Basilica of Bom Jesus','Bekal Fort','Charminar','City Palace','Elephanta Cave','Fatehpur Sikri','Gateway of India','Gingee Fort','Golden Temple','Golkonda Fort','Gwalior Fort','Hawa Mahal','Hill Palace','Howrah Bridge','Humayuns Tomb','India Gate','Jama Masjid','Janta Mantir','Kaye Monastry','Konark Sun Temple','Lotus Temple','Madurai Meenakshi Temple','Mysore Palace','Nalanda University','Qutub Minar','Ran ki Vav','Rashtrapati Bhavan','Red Fort','Sanchi Stupa','Shore Temple Mahabalipuram','Taj Mahal','Thanjavur Chola Temple','Victoria Memorial','Victoria Terminal','Vidhana Soudha','Vivekananda Rock Memorial']
            tour= st.selectbox('Please Choose the location to continue chatting', labels )
            name = "Text_Data/"+tour+".txt"
            with open(name, mode="r",encoding="utf-8" ) as input_file:
                passage = input_file.read()
            chat(passage)
             
    elif choice =="About":
        webbrowser.open_new_tab("about.html")
         
if __name__ == '__main__':
    main()
