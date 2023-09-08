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
from flask import session


model = load_model('Management\model_manage.h5')
intents = json.loads(open('Management\intents_management.json').read())
words = pickle.load(open('Management/texts.pkl','rb'))
classes = pickle.load(open('Management/labels.pkl','rb'))

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
        with open("Emotional_support\dialogue_manage.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_dialogue_to_json(dialogue):
    with open("Emotional_support\dialogue_manage.json", "w") as file:
        json.dump(dialogue, file, indent=4)

def chatbot_response_manage(msg):
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

def task_tracking(msg):

    if msg:
        print("from management.py")
        session['step'] = "task"
        return "What task would you like to track or manage"

    if session["step"] == "task":
        session["step"] = "desc"
        return "Could you please provide a brief description of the task ?"

    if session["step"] == "desc":
        session['step'] = "level"
        return " How important is this task? High, medium, or low priority?"

    if session["step"] == "level":
        session['step'] = "date"
        return "When is the task due or when would you like to complete it?"

    if session["step"] == "date":
        session['step'] = "reminder"
        return "Would you like me to set up reminders for this task? If so, how often should I remind you (daily, weekly, etc.)"
    
    if session["step"] == "reminder":
        session['step'] = "confirm"
        return "Your task has been successfully saved! Is there anything else I can assist you with?"
