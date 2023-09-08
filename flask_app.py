from flask import Flask, render_template, request, session, redirect, url_for, flash
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
from app import chatbot_response
from Emotional_support.emotional_suppport import chatbot_response_emotional, get_random_quote
from Management.management import chatbot_response_manage, task_tracking
from Recommendation.recommendation import get_movie_recommendations
from Healthcare.healthcare import *
app = Flask(__name__)
app.secret_key = "key"

context = {}
nlp = spacy.load("en_core_web_sm")

# Load users from JSON file or create an empty dict
try:
    with open('users.json', 'r') as f:
        users = json.load(f)
except FileNotFoundError:
    users = {}

# Sign In and Sign Up
# Begin
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Hash the password before storing it
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        users[username] = hashed_password.decode('utf-8')

        # Save users to JSON file
        with open('users.json', 'w') as f:
            json.dump(users, f,indent=4)

        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username].encode('utf-8')):
            return redirect(url_for('index'))
        else:
            session['login_error'] = 'Invalid username or password'
            return redirect(url_for('login')) 

    login_error = session.pop('login_error', None)
    return render_template('login.html')

# END


context = {}  # Initialize the context dictionary

@app.route("/healthcare")
def get_bot_healthcare():
    userText = request.args.get('msg')
 
    return predict(userText)
    
    '''
    global context
    page = request.path

    if "predict" in userText.lower():
        context["step"] = "user_age"
        return "Oh, so sad! How old are you?"

    print(context)

    if context.get("step") == "user_age":
        context["user_age"] = int(userText)
        context["step"] = "user_gender"
        return "Got it! Can you specify your gender?"

    if context.get("step") == "user_gender":
        context["user_gender"] = userText
        context["step"] = "main_symptom"
        return "Can you tell me your main symptom?"

    if context.get("step") == "main_symptom":
        main_symptom = preprocess_text(userText)
        most_similar_symptom = find_most_similar_symptom(main_symptom, all_medical_symptoms)
        context["main_symptom"] = most_similar_symptom
        context["step"] = "confirm_main_symptom"
        return f"Are you experiencing   '{most_similar_symptom}'. Is that correct? (yes/no)"

    if context.get("step") == "confirm_main_symptom":
        if userText.lower() == 'yes':
            context["step"] = "add_symptoms"
            return "Do you have any other symptoms? (yes/no)"
        elif userText.lower() == 'no':
            symptoms = [context["main_symptom"]]  # Only the main symptom
            malady_prediction = predict_malady(symptoms)
            context.pop("main_symptom", None)  # Clear only symptom-related information
            return f"Based on the symptoms provided, you may have: {malady_prediction}"
        else:
            return "Please answer with 'yes' or 'no'. Is the main symptom correct? (yes/no)"

    if context.get("step") == "add_symptoms":
        if userText.lower() == 'yes':
            context["step"] = "add_symptom"
            return "Please specify your additional symptom."
        elif userText.lower() == 'no':
            symptoms = [context["main_symptom"]]  # Only the main symptom
            malady_prediction = predict_malady(symptoms)
            context.pop("main_symptom", None)  # Clear only symptom-related information
            return f"Based on the symptoms provided, you may have: {malady_prediction}"
        else:
            return "Please answer with 'yes' or 'no'. Do you have any other symptoms? (yes/no)"

    if context.get("step") == "add_symptom":
        additional_symptom = preprocess_text(userText)
        symptoms = [context["main_symptom"], additional_symptom]
        malady_prediction = predict_malady(symptoms)
        context.pop("main_symptom", None)  # Clear only symptom-related information
        context["step"] = "add_symptoms"  # Allow for more symptoms to be added
        return f"Based on the symptoms provided, you may have: {malady_prediction}"
    '''
    #return chatbot_response(userText)
 # You should define the chatbot_response function



@app.route("/emotional")
def get_bot_emotional():
    userText = request.args.get('msg')
    page = request.path
    print(page)

    # Check if the user is asking for a quote
    if "quote" in userText.lower():
        print("heey from quote")
        return get_random_quote()
    
    return chatbot_response_emotional(userText)

# Initialize the global context dictionary
context = {}

@app.route("/manage")
def get_bot_manage():
    userText = request.args.get('msg')
    global context
    page = request.path
    # Regular chatbot response
    print(page)

    if "task tracking" in userText.lower():
        print("from task tracking!!!!")
        context['step'] = "task"
        return "What task would you like to track or manage"
        
    print(context)
    if context.get("step") == "task":
        context["step"] = "desc"
        return "Could you please provide a brief description of the task?"

    if context.get("step") == "desc":
        context['step'] = "level"
        return "How important is this task? High, medium, or low priority?"

    if context.get("step") == "level":
        context['step'] = "date"
        return "When is the task due or when would you like to complete it?"

    if context.get("step") == "date":
        context['step'] = "reminder"
        return "Would you like me to set up reminders for this task? If so, how often should I remind you (daily, weekly, etc.)"
    
    if context.get("step") == "reminder":
        context['step'] = "confirm"
        return "Your task has been successfully saved! Is there anything else I can assist you with?"

    return chatbot_response_manage(userText)


@app.route("/recommendation")
def get_bot_recommendation():
    userText = request.args.get('msg')
    
    global context
    page = request.path
    print(page)

    if "movie" in userText.lower():
        # Reset context to start a new recommendation request
        context["stage"] = "genre"
        return "Sure! Let's start with your preferred movie genre. What genre are you interested in?"

    if context.get("stage") == "genre":
        # User provided a genre, ask about year
        context["genre"] = userText
        context["stage"] = "year"
        return "Great choice! Now, could you please specify the release year you prefer?"

    if context.get("stage") == "year":
        # User provided a year, ask about runtime
        context["year"] = userText
        context["stage"] = "runtime"
        return "Got it! Lastly, could you tell me your preferred movie runtime in minutes?"

    if context.get("stage") == "runtime":
        # User provided runtime, generate recommendations
        context["runtime"] = userText
        recommendations = get_movie_recommendations(context["genre"], context["runtime"], context["year"])
        
        print(recommendations)
        # Reset context for the next recommendation request
        context = {}


            # Unpack the recommendations and taglines
        recommended_movies = recommendations[0]
        recommended_taglines = recommendations[1]

        # Prepare the recommendation response as a list with HTML line breaks
        recommendation_response = "Here are some recommended movies:<br>"
        for idx, (movie_name, tagline) in enumerate(zip(recommended_movies, recommended_taglines), start=1):
            recommendation_response += f"{idx}. {movie_name} - {tagline}<br>"

        return recommendation_response


    return chatbot_response(userText)


@app.route("/")
def healthcare_page():
    return render_template("healthcare_page.html", current_page='healthcare_page')

@app.route("/emotionnal_support")
def emotionnal_support():
    return render_template("emotionnal_support.html", quote="",current_page='emotionnal_support')

@app.route("/recommendation_page")
def recommendation_page():
    return render_template("recommendation_page.html",current_page='recommendation_page')

@app.route("/management")
def management():    
    return render_template("management_page.html",current_page='management_page')

if __name__ == "__main__":
    app.run(debug=True)

