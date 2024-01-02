# Name: Dong Han
# Mail: dongh@mun.ca
import json

import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

from joblib import dump

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def preprocess_data(grocery_dataset):
    # Load the .json data
    with open(grocery_dataset, 'r', encoding='utf-8') as f:
        grocery_data = json.load(f)

    # Convert json to df
    df = pd.json_normalize(grocery_data)

    # Preprocess the text data
    df["Processed_Name"] = df["name"].apply(preprocess_text)

    # if it's a list
    # Or flatter it:
    # df['category'] = df['category'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    # or Use first as the primary category:
    df['category'] = df['category'].apply(lambda x: x[0] if isinstance(x, list) else x)

    return df

def save_model(model):
    dump(model,"grocery_classify_model.joblib")

def grocery_class_model(df):
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['Processed_Name'], df['category'], test_size=0.33)

    # TF-IDF Vectorizer and Naive Bayes Classifier
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # param grid
    param_grid = {
        "tfidfvectorizer__max_df":[0.5, 0.75, 0.95],
        "tfidfvectorizer__min_df":[0.01, 0.05, 0.1],
        "tfidfvectorizer__ngram_range":[(1, 1), (1, 2),(1,3)],
        "multinomialnb__alpha":[0.01, 0.1, 1]
    }

    # gridsearchcv trainning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best param and model
    print("Best parameters found: ", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    save_model(best_model)

    # Predict and Evaluate
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred,zero_division=0))

def main():
    grocery_dataset = "grocery_data.json"
    df = preprocess_data(grocery_dataset)
    grocery_class_model(df)

if __name__ == '__main__':
    main()