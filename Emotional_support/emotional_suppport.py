import openai 
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import spacy
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
from textblob import TextBlob 
from datetime import datetime
import requests
import bcrypt
from sklearn.ensemble import RandomForestClassifier


model = load_model('Emotional_support\model_emotional.h5')
intents = json.loads(open('Emotional_support\intents_mental.json').read())
words = pickle.load(open('Emotional_support/texts.pkl','rb'))
classes = pickle.load(open('Emotional_support/labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

dialogue = []
def load_dialogue_from_json():
    try:
        with open("Emotional_support\dialogue_emotional.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_dialogue_to_json(dialogue):
    with open("Emotional_support\dialogue_emotional.json", "w") as file:
        json.dump(dialogue, file, indent=4)

def chatbot_response_emotional(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    interaction = {
        "timestamp": timestamp,
        "user_input": msg, 
        "bot_response": res,
        "tag": "tag"
    }

    dialogue = load_dialogue_from_json()
    dialogue.append(interaction)
    save_dialogue_to_json(dialogue)

    return res

# Get Quote
def get_random_quote():
    df = pd.read_csv('Quotes2.csv')
    quotes = df['Quote'].tolist()
    print("from emotion")
    return random.choice(quotes)
