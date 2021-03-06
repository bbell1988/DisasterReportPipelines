import sys

from sqlalchemy import create_engine

import pandas as pd
import numpy as np

def load_data(messages_filepath, categories_filepath):
    
    """
    This function takes two filepaths, reads in csv files, and merges them into 1 dataframe
    
    Args:
        messages_filepath
        categories_filepath
    Returns:
        Dataframe with two csv files merged
        
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = pd.merge(messages, categories, how = 'inner', on = 'id')
    
    return df

def clean_data(df):

    """ 
    This function takes a dataframe and expands the 'categories' column, assigning each feature
    numerical values of 0 for False and 1 for True.
    
    Args:
        A dataframe with a 'categories' column
    
    Returns:
        A cleaned dataframe with 'categories' column expanded and categorical values of 0 or 1
    
    """
    
    # Split 'categories' into separate category columns.
    categories = df['categories'].str.split(pat=';', expand = True)

    # Select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # Remove the last two characters from the 'categories' in the 'row' series
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename the 'categories' columns
    categories.columns = category_colnames

    #Convert category values to 0 or 1.
    for column in categories:
    
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Drop the original categories column from 'df'
    df = df.drop('categories', axis=1)

    # Concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Removing rows with value of 2 in the 'related' column
    df = df[df['related'] != 2]
    
    return df

def save_data(df, database_filename):
    
    """
    This function takes in a dataframe and returns a sql database
    
    Args: 
        df
        database_filename
        
    Returns: 
        A database file called 'DisasterResponse.db'
        
    """
    
    # Save the clean dataset into an sqlite database.
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()