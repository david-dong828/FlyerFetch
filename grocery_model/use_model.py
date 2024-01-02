# Name: Dong Han
# Mail: dongh@mun.ca

# To implement the trained model to classify the saved clean grocery data

from joblib import load
import pandas as pd
from train_grocery_classify_model import preprocess_text

def classify_grocery_data(dataFile,modelFile):
    model = load(modelFile)
    df = pd.read_csv(dataFile)

    df["processed_name"] = df["name"].apply(preprocess_text)

    predicated_categories = model.predict(df["processed_name"])
    df["category"] = predicated_categories
    df.to_csv("withcategory.csv",index=False)
    print("Added categories successfully !")

if __name__ == '__main__':
    dataFile = "../cleaned_data/cleaned_sobeysFlyer_2024-01-01.csv"
    modelFile = "grocery_classify_model.joblib"
    classify_grocery_data(dataFile,modelFile)