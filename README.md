# Disaster Response Pipelines

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Project Description](#description)
4. [File Descriptions](#files)
5. [Instructions](#instructions)

## Installation <a name="installation"></a>

All necessary libraries are available in the Anaconda distribution of Python. The libraries used in this project include:

- pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

The code is compatible with Python versions 3.x.

## Project Motivation <a name="motivation"></a>

The goal of this project is to classify disaster messages into various categories. By analyzing disaster data provided by [Figure Eight](https://www.figure-eight.com/), a model was built for an API that can classify disaster messages. Users can input a new message through a web application and receive classification results across multiple categories. The web app also provides data visualizations.

## Project Description <a name="description"></a>

The project consists of three main components:

## ETL Pipeline

The `process_data.py` script implements an ETL pipeline that performs the following steps:

1. **Load Data**:
   - Reads datasets from CSV files:
     - `messages_filepath`: Path to the messages CSV file.
     - `categories_filepath`: Path to the categories CSV file.
   - Merges the two datasets based on the `id` column.

2. **Clean Data**:
   - Splits the `categories` column into individual category columns.
   - Converts category values from strings to numeric format.
   - Removes rows where the `related` category has a value of 2, as this value is not meaningful.
   - Eliminates duplicate rows from the dataset.

3. **Save data**:
   - Saves the cleaned dataset to an SQLite database.

## ML Pipeline

The `train_classifier.py` script implements a machine learning pipeline that:
- **Loads** data from a SQLite database. The `load_data` function retrieves feature data (text messages) and target variables from the `disaster_messages_tbl` table, along with category names.
- **Splits** the data into training and test sets using the `train_test_split` function to ensure robust evaluation of the model's performance.
- **Builds** a text processing and machine learning pipeline. The `build_model` function constructs a pipeline that includes:
  - **Tokenization and Normalization:** Using the `CountVectorizer` with a custom `tokenize` function that normalizes, tokenizes, and lemmatizes the text data.
  - **TF-IDF Transformation:** Applying `TfidfTransformer` to convert token counts into normalized term frequency-inverse document frequency (TF-IDF) representation.
  - **Multi-output Classification:** Utilizing `MultiOutputClassifier` with `AdaBoostClassifier` to handle multi-label classification.
- **Trains** and **Tunes** the model using `GridSearchCV` to optimize hyperparameters such as the number of boosting stages and learning rate for AdaBoost.
- **Evaluates** the model on the test set, printing classification metrics for each category and overall accuracy.
- **Saves** the trained model as a pickle file using the `save_model` function, which allows for future use and deployment.


3. **Flask Web App:** The web app allows users to enter a disaster message and view its categorized results. It also includes visualizations that describe the data.

## File Descriptions <a name="files"></a>

The project files are organized as follows:

- **README.md:** This file provides an overview and instructions for the project.
- **ETL Pipeline Preparation.ipynb:** Contains code for preparing the ETL pipeline.
- **ML Pipeline Preparation.ipynb:** Contains code for preparing the ML pipeline.
- **workspace/**
  - **app/**
    - **run.py:** Flask application to run the web app.
    - **templates/**
      - **master.html:** Main page of the web application.
      - **go.html:** Result page displaying the classification results.
  - **data/**
    - **disaster_categories.csv:** Dataset containing message categories.
    - **disaster_messages.csv:** Dataset containing disaster messages.
    - **DisasterResponse.db:** SQLite database storing processed data.
    - **process_data.py:** Script for the ETL process.
  - **models/**
    - **train_classifier.py:** Script for training the classification model.

## Instructions <a name="instructions"></a>

To run the app, follow these steps:

1. Set up the database and model by running the following commands from the project’s root directory:

    - To run the ETL pipeline that cleans the data and stores it in the database:
      ```bash
      python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
      ```
    - To run the ML pipeline that trains the classifier and saves the model:
      ```bash
      python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
      ```

2. Run the following command in the `app` directory to start the web application:
    ```bash
    python run.py
    ```

3. Open a browser and go to `http://0.0.0.0:3001/` to interact with the app.
