import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes and lemmatizes the input text.

    Args:
    text (str): Input text to be tokenized and lemmatized.

    Returns:
    clean_tokens (list): List of cleaned tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data
engine = create_engine('sqlite:///../data/InsertDatabaseName.db')
df = pd.read_sql_table('messages_categories', engine)

# Load model
model = joblib.load("../models/classifier.pkl")

# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Visualization 1: Distribution of Message Genres
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # Visualization 2: Distribution of Message Categories
    category_names = df.columns[4:]
    category_counts = df[category_names].sum().sort_values(ascending=False)

    graphs.append({
        'data': [
            Bar(
                x=category_names,
                y=category_counts
            )
        ],
        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Categories"
            }
        }
    })

    # Visualization 3: Top 10 Message Categories
    top_10_categories = category_counts.head(10)
    top_10_category_names = list(top_10_categories.index)

    graphs.append({
        'data': [
            Bar(
                x=top_10_category_names,
                y=top_10_categories
            )
        ],
        'layout': {
            'title': 'Top 10 Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Categories"
            }
        }
    })

    # Encode Plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with Plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html page with the classification results
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
