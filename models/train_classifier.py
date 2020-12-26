import sys
import pandas as pd 
import numpy as np 
from sqlalchemy import create_engine
import re
import pickle

import nltk
nltk.download(['punkt','stopwords','wordnet'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """Loads and merges the messages and categories databases
    
    Args: database_filepath
    
    Returns:
        X: Dataframe of features
        Y: Dataframe of labels
        category_names: Column names of the labels   
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # Dropping rows that have a value of 2
    df.drop(df[df['related'] == 2].index, inplace = True) 
                           
    # Creating the X and Y datasets
    X = df.loc[:,'message']
    Y = df.iloc[:,4:]
    
    # Creating list of category_names
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize
    words = word_tokenize(text)
    
    # Stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # Lemmatizing
    stop_words = stopwords.words("english")
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
   
    return words_lemmed


def build_model():
    
        model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 100,
                                                            min_samples_split = 10,
                                                            min_samples_leaf = 3,
                                                            max_depth = None)))
        ])
        
        parameters = {'clf__estimator__n_estimators': [10,50,100],
                      'clf__estimator__max_depth': [None, 3, 7, 10],
                      'clf__estimator__min_samples_split': [2, 5, 10],
                      'clf__estimator__min_samples_leaf':[1,3,10]} 
        
        model = RandomizedSearchCV(model,
                                      param_distributions = parameters,
                                      cv = 3,
                                      n_jobs = -1,
                                      verbose = 2)
    
        return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names = category_names))

   
def save_model(model, model_filepath):
   
    pkl_filename = '{}'.format(model_filepath)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


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