# Used to check groceries' flyers
import csv,os

from flask import Flask,request,render_template,jsonify,Response
import getFlyer_sobeys_walmart,data_clean
from grocery_model import train_recomdation_model, useBiLSTMmodel
from itertools import groupby
import json
from datetime import datetime

app = Flask(__name__,static_folder='assets')

def sobeys_walmart_flyer(url):
    draft_flyer_file,shopName = getFlyer_sobeys_walmart.getFlyer(url)
    if draft_flyer_file == -1:
        print("error in getting flyer data")
        return
    cleaned_filePath = data_clean.clean_data(draft_flyer_file,shopName)
    return cleaned_filePath

def generate_html_table(data, headers):
    table = "<table class='table'>"
    table += "<thead><tr><th>" + "</th><th>".join(headers) + "</th></tr></thead>"
    table += "<tbody>"
    for category, rows in data:
        table += f"<tr><td colspan='{len(headers)}'><strong>{category}</strong></td></tr>"
        for row in rows:
            table += "<tr><td>" + "</td><td>".join(row[header] for header in headers) + "</td></tr>"
    table += "</tbody></table>"
    return table

def read_csv(filePath):
    with open(filePath, 'r', newline='', encoding="utf-8") as file:
        reader = list(csv.DictReader(file))

        # Create separate lists for each store
        sobeys_data = []
        walmart_data = []

        # Populate the lists, ensuring that rows with empty store-specific data are not added
        for row in reader:
            if any(row[key].strip() for key in row.keys() if 'Sobeys' in key):
                sobeys_data.append(row)
            if any(row[key].strip() for key in row.keys() if 'Walmart' in key):
                walmart_data.append(row)

        sorted_sobeys = sorted(sobeys_data, key=lambda x: x['Sobeys_predict_category'])
        sorted_walmart = sorted(walmart_data, key=lambda x: x['Walmart_predict_category'])

        # Group by category
        grouped_sobeys = groupby(sorted_sobeys, key=lambda x: x['Sobeys_predict_category'])
        grouped_walmart = groupby(sorted_walmart, key=lambda x: x['Walmart_predict_category'])

        # Generate HTML tables
        sobeys_headers = [header for header in reader[0].keys() if 'Sobeys' in header and 'category' not in header]
        walmart_headers = [header for header in reader[0].keys() if 'Walmart' in header and 'category' not in header]

        sobeys_table = generate_html_table(grouped_sobeys, sobeys_headers)
        walmart_table = generate_html_table(grouped_walmart, walmart_headers)

        return sobeys_table,walmart_table

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/flyers",methods=['GET'])
def showFlyers():
    folder_path = 'combined_data'
    fileName = "combined_" + datetime.now().strftime("%Y-%m-%d") + ".csv"
    filePath = os.path.join(folder_path, fileName)

    if not os.path.exists(filePath):
        cleanedFilesPaths = []
        urls = ["https://www.sobeys.com/en/flyer/", "https://www.walmart.ca/en/flyer"]
        for url in urls:
            cleaned_filePath = sobeys_walmart_flyer(url) #get cleaned data path back
            cleanedFilesPaths.append(cleaned_filePath)

        data_clean.combine_data(cleanedFilesPaths[0],cleanedFilesPaths[1])

        # train_recomdation_model.get_recommendation_simple(cleanedFilesPaths[0]) # get sobeys/ walmart SIMPLE recomemndations
        # train_recomdation_model.get_recommendation_simple(cleanedFilesPaths[1]) # so no need to run in showRecommendations()

        # useBiLSTMmodel.add_predicted_categories(cleanedFilesPaths[0]) # Add predicted categories and subcategories
        # useBiLSTMmodel.add_predicted_categories(cleanedFilesPaths[1]) # for both Sobeys n Walmart. Save to recommendation folder

    print(f"done {filePath}")
    sobeys_table, walmart_table = read_csv(filePath)

    return jsonify(sobeys=sobeys_table, walmart=walmart_table)

def jsnonTolist(recommendationJson):
    recommendationList = []
    for key, items in recommendationJson.items():
        print(items)
        new_item = {
            "name": items["Item_Name"],
            "price": float(items["Price"].replace("$", "")),  # Convert price to float and remove '$'
            "uom": items["UoM"],
            "category":items["Category"]

        }
        recommendationList.append(new_item)
    return recommendationList

@app.route("/recommendations",methods=['POST'])
def showRecommendations():
    selected_categories = request.json['categories']
    # print(selected_categories)
    sobeys_recommendations_json,walmart_recommendations_json = train_recomdation_model.get_recommendations(selected_categories)
    # print("sobeys_recommendations_json ",sobeys_recommendations_json)

    if sobeys_recommendations_json != -1:
        sobeys_recommendations = jsnonTolist(sobeys_recommendations_json)
        walmart_recommendations = jsnonTolist(walmart_recommendations_json)

        return jsonify(sobeys=sobeys_recommendations, walmart=walmart_recommendations)

def main():
    # urls = ["https://www.sobeys.com/en/flyer/", "https://www.walmart.ca/en/flyer"]
    # for url in urls:
    #     sobeys_walmart_flyer(url)
    # showFlyers()
    showRecommendations()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # main()

