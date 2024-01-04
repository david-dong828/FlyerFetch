# Used to check groceries' flyers
import csv,os

from flask import Flask,request,render_template,jsonify,Response
import getFlyer_sobeys_walmart,data_clean
from grocery_model import train_recomdation_model
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

        sorted_sobeys = sorted(sobeys_data, key=lambda x: x['Sobeys_category'])
        sorted_walmart = sorted(walmart_data, key=lambda x: x['Walmart_category'])

        # Group by category
        grouped_sobeys = groupby(sorted_sobeys, key=lambda x: x['Sobeys_category'])
        grouped_walmart = groupby(sorted_walmart, key=lambda x: x['Walmart_category'])

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
            cleaned_filePath = sobeys_walmart_flyer(url)
            cleanedFilesPaths.append(cleaned_filePath)

        data_clean.combine_data(cleanedFilesPaths[0],cleanedFilesPaths[1])

        train_recomdation_model.get_recommendation_simple(cleanedFilesPaths[0]) # get sobeys/ walmart recomemndations
        train_recomdation_model.get_recommendation_simple(cleanedFilesPaths[1]) # so no need to run in showRecommendations()

    print(f"done {filePath}")
    sobeys_table, walmart_table = read_csv(filePath)

    return jsonify(sobeys=sobeys_table, walmart=walmart_table)

@app.route("/recommendations",methods=['GET'])
def showRecommendations():
    folder_path = 'recommendation'
    fileName1 = "walmart_recommendations_" + datetime.now().strftime("%Y-%m-%d") + ".json"
    filePath1 = os.path.join(folder_path, fileName1)
    fileName2 = "sobeys_recommendations_" + datetime.now().strftime("%Y-%m-%d") + ".json"
    filePath2 = os.path.join(folder_path, fileName2)

    walmart_recommendations = []
    sobeys_recommendations = []

    # Load Walmart recommendations if file exists
    if os.path.exists(filePath1):
        try:
            with open(filePath1, 'r', encoding='utf-8') as file:
                walmart_json = json.load(file)
                for key,items in walmart_json.items():
                    print(items)
                    new_item = {
                        'name': items['Item_Name'],
                        'price': float(items['Price'].replace('$', '')),  # Convert price to float and remove '$'
                        'uom': items['UoM']
                    }
                    walmart_recommendations.append(new_item)
            print(walmart_recommendations)
        except json.JSONDecodeError:
            print(f"Error reading {filePath1}")

    # Load Sobeys recommendations if file exists
    if os.path.exists(filePath2):
        try:
            with open(filePath2, 'r', encoding='utf-8') as file:
                sobeys_json = json.load(file)
                for key,items in sobeys_json.items():
                    new_item = {
                        'name': items['Item_Name'],
                        'price': float(items['Price'].replace('$', '')),  # Convert price to float and remove '$'
                        'uom': items['UoM']
                    }
                    sobeys_recommendations.append(new_item)
        except json.JSONDecodeError:
            print(f"Error reading {filePath2}")

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

