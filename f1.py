import streamlit as st
import pandas as pd
import speech_recognition as sr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write("You said: ", text)
            return text
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            st.error("Could not request results from Google Speech Recognition service; check your network connection")
            return None

@st.cache_data
def load_data():
    data = pd.read_excel('train.xlsx')
    return data

@st.cache_data
def train_model(data):
    X = data['News Headline']
    y = data['Sentiment']
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, max_df=0.9)
    X_train_tfidf = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y, test_size=0.2, random_state=42)
    parameters = {'C': [1, 10], 'gamma': [0.1]}
    svc = SVC(kernel='rbf')
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(X_train, y_train)
    return clf, vectorizer, X_test, y_test, y_train

def update_input():
    recognized_text = recognize_speech()
    if recognized_text is not None:
        st.session_state.user_input1 = recognized_text

clf, vectorizer, X_test, y_test, y_train = train_model(load_data())
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

st.title('Sentiment Analysis Of Financial News')
st.write(f'Model accuracy after tuning: {accuracy:.2f}%')

col1, col2 = st.columns([4, 1])
with col1:
    # Ensure that the value for 'user_input1' is initialized in the session state
    if 'user_input1' not in st.session_state:
        st.session_state.user_input1 = ""
    user_input = st.text_input('Enter a headline or press the microphone button:', key="user_input1")
with col2:
    st.button('Speak', on_click=update_input)

if st.button('Predict') and user_input:
    user_input_tfidf = vectorizer.transform([user_input])
    prediction = clf.predict(user_input_tfidf)
    if prediction[0] == 'positive':
        sentiment = 'Positive'
        st.success('Sentiment: Positive')
    elif prediction[0] == 'negative':
        sentiment = 'Negative'
        st.error('Sentiment: Negative')
    else:
        sentiment = 'Neutral'
        st.info('Sentiment: Neutral')
