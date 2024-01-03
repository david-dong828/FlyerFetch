# Name: Dong Han
# Mail: dongh@mun.ca

# To implement the trained model to classify the saved clean grocery data

from joblib import load
import pandas as pd
from grocery_model import train_grocery_classify_model

def classify_grocery_data(dataFile,modelFile="grocery_model/grocery_classify_model.joblib"):
    model = load(modelFile)
    df = pd.read_csv(dataFile)

    df["processed_name"] = df["Item_Name"].apply(train_grocery_classify_model.preprocess_text)

    predicated_categories = model.predict(df["processed_name"])
    df["category"] = predicated_categories

    df.to_csv(dataFile,index=False) #save back
    print("Added categories successfully !")

if __name__ == '__main__':
    dataFile = "../cleaned_data/cleaned_sobeysFlyer_2024-01-01.csv"

    modelFile = "grocery_classify_model.joblib"
    classify_grocery_data(dataFile,modelFile)