#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge datasets from specified file paths.

    Args:
    messages_filepath (str): Path to the messages CSV file.
    categories_filepath (str): Path to the categories CSV file.

    Returns:
    DataFrame: Merged DataFrame containing messages and categories.
    """
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on 'id'
    df = pd.merge(messages, categories, on='id')
    
    return df
    
def clean_data(df):
    """
    Clean and structure the merged DataFrame by processing category data.

    Args:
    df (DataFrame): Merged DataFrame.

    Returns:
    DataFrame: Cleaned DataFrame with categories split into individual columns.
    """
    # Split 'categories' into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names from the first row
    category_colnames = categories.iloc[0].apply(lambda x: x.split('-')[0]).tolist()
    categories.columns = category_colnames
    
    # Convert category values to numbers
    for column in categories:
        # Take the last character of the string and convert it to integer
        categories[column] = categories[column].str[-1].astype(int)
        
        # Ensure values are binary (0 or 1)
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)
    
    # Drop the original 'categories' column from df
    df = df.drop('categories', axis=1)
    
    # Concatenate the original df with the cleaned categories
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle NaN values by filling them with 0 in the category columns
    df[category_colnames] = df[category_colnames].fillna(0)
    
    # Replace infinity values with NaN and then fill with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to an SQLite database.

    Args:
    df (DataFrame): DataFrame to save.
    database_filename (str): SQLite database filename.
    """
    # Create SQLite engine and save the DataFrame to the database
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages_categories', engine, index=False, if_exists='replace')

def main():
    messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

    print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)
    
    print(f'Saving data...\n    DATABASE: {database_filepath}')
    save_data(df, database_filepath)
    
    print('Data processing complete. Cleaned data has been saved to the database!')

if __name__ == '__main__':
    main()
