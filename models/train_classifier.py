import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pickle

# Download necessary NLTK data
nltk.download('punkt')       # Download the 'punkt' tokenizer data
nltk.download('wordnet')     # Download the 'wordnet' corpus for lemmatization
nltk.download('stopwords')   # Download stopwords if you plan to use them

def load_data(database_filepath):
    """
    Load data from the SQLite database.

    Args:
    database_filepath (str): Path to the SQLite database file.

    Returns:
    X (DataFrame): DataFrame containing feature data (e.g., text messages).
    Y (DataFrame): DataFrame containing target variables.
    category_names (list): List of category names for the target variables.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    try:
        df = pd.read_sql_table('messages_categories', engine)
    except ValueError as e:
        print(f"Error: {e}")
        print("Ensure that the 'messages_categories' table exists in the database.")
        return None, None, None
    
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    
    return X, Y

def tokenize(text):
    """
    Tokenize, normalize, and lemmatize the input text.

    Args:
    text (str): The text message to process.

    Returns:
    tokens (list of str): List of processed tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]
    return tokens

def build_model():
    """
    Construct a machine learning pipeline and configure Grid Search for hyperparameter tuning.
    
    Returns:
    cv (GridSearchCV): The GridSearchCV object set up with the pipeline and parameter grid.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [50, 70],
        'clf__estimator__learning_rate': [0.1]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=3, n_jobs=-1)

    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the model and print classification metrics.

    Args:
    model: Trained machine learning model.
    X_test (DataFrame): Test set of features.
    Y_test (DataFrame): Test set of target variables.
    category_names (list): List of category names for the target variables.
    """
    Y_pred = model.predict(X_test)
    
    for i, category in enumerate(Y_test.columns):
        print(f'Category: {category}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
    
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print(f'Overall accuracy: {overall_accuracy:.4f}')

def save_model(model, model_filepath):
    """
    Save the model as a pickle file.

    Args:
    model: Trained machine learning model.
    model_filepath (str): File path where the model will be saved.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)
        
        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)
        
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
