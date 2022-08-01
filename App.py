# Core Pkgs
import streamlit as st 
st.set_page_config(layout="wide")


# EDA Pkgs
import pandas as pd 
import numpy as np 
from nltk.corpus import stopwords
import re

# Utils
import joblib 
pipe_lr = joblib.load(open("Capstone_31_07.pkl","rb"))

header = st.container()
model_prediction = st.container()

with header:
    
    st.markdown("<h1 style='text-align: center; color: white;'>Capstone_AIML_JPMC_G3</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>Automatic Ticket Assignment using NLP techniques </h2>", unsafe_allow_html=True)
    


with st.form("my_form"):
    st.write("Ticket details")
    short_desc = st.text_input('Plesase provide a short description of your issue')
    long_desc = st.text_input('Plesase provide the description of your issue')

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        short_desc = [short_desc]
        pred = pipe_lr.predict(short_desc)
        pred_prob = pipe_lr.predict_proba(short_desc)
        st.write('The ticket is assigned to :',pred[0] )
        st.write('Probability given by model:',pred_prob.max() )
    
    






 









