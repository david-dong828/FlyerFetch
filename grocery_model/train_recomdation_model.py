# Name: Dong Han
# Mail: dongh@mun.ca

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
import json,os
from datetime import datetime

def preprocess_data(flyerData):
    df = pd.read_csv(flyerData)

    # TF-IDF Vectorization of item names
    vectorizer = TfidfVectorizer(stop_words='english')
    df['item_name_vectors'] = list(vectorizer.fit_transform(df['Item_Name']).toarray())

    # Price normalization
    df["Price"] = pd.to_numeric(df["Price"],errors="coerce")
    # prices = df['Price'].values.reshape(-1, 1)
    scalar = MinMaxScaler()
    df["normlized_price"] = scalar.fit_transform(df[['Price']].fillna(df['Price'].mean()).values)

    return df,vectorizer

def get_recommendations(df,vectorizer):
    recommendations = {}

    # similarity_matrix = calculate_similarity(df["item_name_vectors"])
    for category in df["category"].unique():
        category_items = df[df["category"] == category]

        # Select the item with the lowest normalized price
        best_item = category_items.loc[category_items['normlized_price'].idxmin()]
        print(best_item)
        recommendations[category] = best_item
    return recommendations

def get_recommendation_simple(flyerData):
    print(flyerData)
    folder_path = 'recommendation'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    today_date = datetime.now().strftime("%Y-%m-%d")
    if "sobeys" in flyerData:
        fileName = "sobeys_" + "recommendations_" + today_date + ".json"
    else: fileName = "walmart_" + "recommendations_" + today_date + ".json"
    filePath = os.path.join(folder_path, fileName)

    if not os.path.exists(filePath):

        recommendations = OrderedDict() # To keep the first added stay at first

        df = pd.read_csv(flyerData)

        # find hot price items
        hot_price_items = df[df["remark"].str.contains("HOT PRICE|Save",na=False)]
        for _, row in hot_price_items.iterrows():
            item_key = row['Item_Name'].lower()  #use low case for avoiding duplication
            if item_key not in recommendations:
                recommendations[item_key] = {
                    'Item_Name': row['Item_Name'],
                    'Price': row['Price'],
                    'UoM': row['UoM']
                }

        # find preference items
        keywords = ['beef', 'pork', 'chicken', 'salmon', 'egg', 'avocado','silk', 'apple', 'bread','yogurt','chip']
        for keyword in keywords:
            keyword_items = df[df['Item_Name'].str.contains(keyword, case=False, na=False)]
            keyword_count = 0
            for _, row in keyword_items.iterrows():
                if keyword_count < 2:
                    item_key = row['Item_Name'].lower()  # Use item name in lowercase as the key to avoid duplicates
                    if item_key not in recommendations:
                        recommendations[item_key] = {
                            'Item_Name': row['Item_Name'],
                            'Price': row['Price'],
                            'UoM': row['UoM']
                        }
                        keyword_count += 1
        with open(filePath, 'w') as file:
            json.dump(recommendations, file, indent=4)

    return filePath

def main():
    flyerData = "../cleaned_data/cleaned_sobeysFlyer_2024-01-03.csv"
    flyerData1 = "../cleaned_data/cleaned_walmartFlyer_2024-01-03.csv"
    # df, v = preprocess_data(flyerData)
    # r = get_recommendations(df,v)
    # print(r)

    r = get_recommendation_simple(flyerData1)
    print(r)
if __name__ == '__main__':
    main()
