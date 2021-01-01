# Disaster Response Pipeline Project

## Table of Contents

1. Installation
2. Project Motivation
3. File Descriptions
4. Results
5. Licenses, Authors, and Acknowledgments

## Installation

The libraries used in this study are sqlalchemy, pandas, numpy, re, sklearn, nltk, pickle, json, plotly, and flask. The code should run with no issues using python 3.

## Project Motivation

In this project, I used the disaster dataset from Appen, formerly Figure Eight. The goal was to create a model that would classify disaster messages into 36 categories, and then deploy the model to a website. An ETL pipeline and ML pipeline needed to be run to complete this study/

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
