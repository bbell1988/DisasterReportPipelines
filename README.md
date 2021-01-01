# Disaster Response Pipeline Project

## Table of Contents:

1. Installation
2. Project Motivation
3. File Descriptions
4. Results
5. Licenses, Authors, and Acknowledgments

## Installation:

The libraries used in this study are sqlalchemy, pandas, numpy, re, sklearn, nltk, pickle, json, plotly, and flask. The code should run with no issues using python 3.

## Project Motivation:

Appen, formerly Figure Eight, has supplied a dataset of pre-labeled tweets and text messages that would be received after real-life disasters. Generally after a disaster, hundreds of thousands of messages will be received, but only 1 in 1000 may actually be relevant to the specific disaster-response team. As a data scientist, I have been tasked with preparing the data in an ETL pipeline and creating a supervised machine learning model that can classify these messages into categories. This will help the various disaster-response organizations more efficiently respond to the correct messages. Following that, I am to create a web app that would allow a worker to input a message, and receive the various categories that the message would apply to.

Below are the instructions for running the files

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Descriptions

1. disaster_categories.csv - csv file of the categories 
2. disaster_messages.csv - csv file of the messages
3. process_data.py - python file that combines the csv files, cleans them, and outputs a database
4. DisasterResponse.db - the database created from the process_data.py file
5. train_classifier.py - python file that creates a model to classify the messages, outputs a pickle file
6. run.py - runs the python web app

## Results:

1. If we look at the distribution of the genre of messages, we can see that the least amount of messages were delivered by social media, and the most were delivered by the news.
2. The top 5 categories were related, aid_related, weather_related, direct_report, and request
3. Nearly 81% of the messages classified were non-direct report. Approx. 19% were direct report.

## Licenses, Authors, and Acknowledgments:

Special thanks to Appen, formerly Figure Eight for the disaster datasets





