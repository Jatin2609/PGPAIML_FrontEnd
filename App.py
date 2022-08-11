# Core Pkgs
import streamlit as st 
st.set_page_config(layout="wide")
# nltk.download('stopwords')

# EDA Pkgs
import pandas as pd 
import numpy as np 
# from nltk.corpus import stopwords
import re

# Utils
import joblib 
pipe_lr = joblib.load(open("Capstone_08_11.pkl","rb"))

header = st.container()
model_prediction = st.container()


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

#define a function for preprocessing the data
def preprocess_text(df, column_name=''):

    # Remove email Ids
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]{2,4}',' ',x))
    # Remove label url link
    df[column_name]=df[column_name].apply(lambda x: re.sub(r'urlLink|urllink','',x))
    # remove all the places where that starts with http or https
    df[column_name]=df[column_name].apply(lambda x: re.sub(r'https?\S+','',x))   
    df[column_name] = df[column_name].apply((lambda x: re.sub(r'([xx]+)|([XX]+)|(\d+)', '',x)))
    df[column_name] = df[column_name].apply((lambda x: re.sub(r'[_D_\n_D_\n]', '',x)))
    # Strip unwanted spaces
    df[column_name] = df[column_name].apply(lambda x: x.strip())
    # Select only alphabets
    df[column_name] = df[column_name].apply(lambda x: re.sub('[^A-Za-z]+', ' ', x))
    # Convert text to lowercase
    df[column_name] = df[column_name].apply(lambda x: x.lower())
    # Remove stopwords
    df[column_name] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    #Remove hello
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'hello |please |received |unable |see |help |received |need |sent |yes |no |na ', ' ', x))
    # Replace empty strings with Null
    df[column_name].replace('', np.nan, inplace = True)
    # Drop Null values
    df = df.dropna()

    return df


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def lemmitize(sentence):
    sentence=str(sentence)
    # sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)  
    # filtered_words = [w for w in tokens if len(w) >= 2 if not w in stopwords.words('english')]
    #stem_words = [stemmer.stem(w) for w in tokens]
    lemma_words = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(lemma_words)

with header:
    
    st.markdown("<h1 style='text-align: center; color: white;'>Capstone_AIML_JPMC_G3</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>Automatic Ticket Assignment using NLP techniques </h2>", unsafe_allow_html=True)
    


with st.form("my_form"):
    st.write("Ticket details")
    short_desc = st.text_input('Plesase provide a short description of your issue')
    long_desc = st.text_input('Plesase provide the description of your issue')
    dfs = pd.DataFrame()
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        dfs.at[0, 'Description'] = long_desc
        dfs.at[0,'Short description'] = short_desc
        # preprocessing both columns
        dfs = preprocess_text(dfs, column_name='Description')
        dfs = preprocess_text(dfs, column_name='Short description')
        # Lemmatizing both columns
        dfs['Short description'] = dfs['Short description'].apply(lambda s:lemmitize(s))
        dfs['Description'] = dfs['Description'].apply(lambda s:lemmitize(s))
        # Concatenating both columns
        for i in range(len(dfs)):
            if(dfs.iloc[i, 0]== dfs.iloc[i, 1]):
                dfs.iloc[i,1]="" 

        dfs["issue_description"]=dfs["Short description"] + " " + dfs["Description"]
        
        pred = pipe_lr.predict(dfs["issue_description"])
        pred_prob = pipe_lr.predict_proba(dfs["issue_description"])
        pred_prob.sort()
        if pred_prob[:,-1]-pred_prob[:,-2] >0.5:
            st.write('The ticket is assigned to :',pred[0] )
            st.write('Probability given by model:',pred_prob.max() )
        else :
            st.write('The ticket cannot be assigned automatically with a reasonable accuracy. Assigning it to manual assignment team' )
        
    
    






 









