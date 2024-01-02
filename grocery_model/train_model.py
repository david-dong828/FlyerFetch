# Name: Dong Han
# Mail: dongh@mun.ca
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)


def grocery_class_model(grocery_dataset):
    # Load the .json data
    with open(grocery_dataset,'r',encoding='utf-8') as f:
        grocery_data = json.load(f)

    # Convert json to df
    df = pd.json_normalize(grocery_data)
    print(df.shape)
    print(df.head())
    print(df.tail())
    # Preprocess the text data
    df["Processed_Name"] = df["name"].apply(preprocess_text)
    # df["Processed_Name"] = df["name"]
    print(df["Processed_Name"].count())
    print(df.tail())
    # Use first as the primary category if it's a list
    # Or flatter it: df['category'] = df['category'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df['category'] = df['category'].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['Processed_Name'], df['category'], test_size=0.2, random_state=42)

    # TF-IDF Vectorizer and Naive Bayes Classifier
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Training the model
    model.fit(X_train, y_train)

    # Predict and Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred,zero_division=0))

def main():
    grocery_dataset = "grocery_data.json"
    grocery_class_model(grocery_dataset)

if __name__ == '__main__':
    main()