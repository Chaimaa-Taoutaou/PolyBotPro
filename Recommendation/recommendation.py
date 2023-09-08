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
movies_data = pd.read_csv("Recommendation\Movies.csv")

# Recommendation
# Begin
# Create a function to generate movie recommendations based on user preferences
def get_movie_recommendations(genre, runtime, year, num_recommendations=5):
    # Combine relevant features into a single string for each movie
    #movies_data["Features"] = movies_data["genre"] + " " + movies_data["run_time"] + " " + movies_data["year"]
    recommendation = []
    # Convert text data into a numerical format using CountVectorizer
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(movies_data["genre"])

    # Calculate the cosine similarity between movies
    cosine_sim = cosine_similarity(count_matrix)

    # Find the index of the selected movie in the dataset
    selected_movie_index = movies_data[movies_data["genre"] == genre].index[0]

    # Get the similarity scores for all movies compared to the selected movie
    similarity_scores = list(enumerate(cosine_sim[selected_movie_index]))

    # Sort movies based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of top similar movies
    top_indices = [i[0] for i in similarity_scores[1:num_recommendations+1]]

    # Get the movie titles for recommendations
    name = movies_data.iloc[top_indices]["name"].tolist()
    tagline = movies_data.iloc[top_indices]["tagline"].tolist()
    recommendation.append(name)
    recommendation.append(tagline)
    return recommendation


# Weather
# Begin
def extract_city_name(text):
    doc = nlp(text)
    city = None
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE stands for Geo-Political Entity (locations)
            city = ent.text
            break
    return city

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        print(data)
        if data["cod"] == 200:
            weather_info = {
                "city": data["name"],
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
            }
            return weather_info
        else:
            print("Error: Unable to fetch weather data.")
            return None

    except requests.exceptions.RequestException as e:
        print("Error: ", e)
        return None


    if weather_data:
        print("\nWeather in {}: ".format(weather_data["city"]))
        print("Temperature: {}°C".format(weather_data["temperature"]))
        print("Description: {}".format(weather_data["description"]))
        print("Humidity: {}%".format(weather_data["humidity"]))
        print("Wind Speed: {} m/s".format(weather_data["wind_speed"]))

# End 