# Name: Dong Han
# Mail: dongh@mun.ca

# To compare the same or similar grocery items between groceries
import pandas as pd
import string
from fuzzywuzzy import process

def preprocess_name(name):
    name = name.lower()
    name = name.translate(str.maketrans('', '', string.punctuation))
    return name

def preprocess_data(dataFile):
    df = pd.read_csv(dataFile)
    df["processed_name"] = df["name"].apply(preprocess_name)
    return df

def fina_match(name,choices, threshold):
    match = process.extractOne(name,choices,score_cutoff=threshold)
    return match[0] if match else None

def comparision(dataFile1, dataFile2):
    data1 = preprocess_data(dataFile1)
    data2 = preprocess_data(dataFile2)

    threshold = 80
    data1["matched_name"] = data1["processed_name"].apply(lambda x: fina_match(x,data2["processed_name"],threshold))
    data1.to_csv("compared.csv",index=False)
    print("saved compared !")

if __name__ == '__main__':
    dataFile1 = "../cleaned_data/cleaned_sobeysFlyer_2024-01-02.csv"
    dataFile2 = "../cleaned_data/cleaned_walmartFlyer_2024-01-02.csv"
    comparision(dataFile1, dataFile2)