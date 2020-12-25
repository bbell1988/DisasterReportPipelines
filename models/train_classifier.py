import sys

# import libraries
from sqlalchemy import create_engine
import pandas as pd 
import numpy as np 
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download(['punkt','stopwords','wordnet'])

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

import pickle


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    df.drop(df[df['related'] == 2].index, inplace = True) 
    X = df.loc[:,'message']
    Y = df.iloc[:,4:]

def tokenize(text):
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words("english")
    
    # Tokenize
    words = word_tokenize(text)
    
    # Stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # Lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
   
    return words_lemmed


def build_model():
    
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state = 10)
    pipeline.fit(X_train, Y_train)
    
    return model, X_test, Y_test


def evaluate_model(model, X_test, Y_test, category_names):
    category_names = df.columns[4:]
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = category_names)
    
    for i, var in enumerate(ylabels):
        print(var)
        print(classification_report(y_test.iloc[:,i],y_pred.iloc[:,i]))

def save_model(model, model_filepath):
    with open('pickle_rCV','wb') as f:
        pickle.dump(random_search, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()