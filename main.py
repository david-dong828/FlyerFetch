# Used to check groceries' flyers
import csv,os

from flask import Flask,request,render_template,jsonify,Response
from markupsafe import Markup
import getFlyer_sobeys_walmart,data_clean
from itertools import groupby

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

# def read_csv(filePath):
#     with open(filePath, 'r', newline='',encoding="utf-8") as file:
#         reader = csv.DictReader(file)
#         sorted_list = sorted(reader, key=lambda x: (x['Sobeys_category'], x['Walmart_category']))
#         headers = [header for header in sorted_list[0].keys() if header not in ['Sobeys_category', 'Walmart_category']]
#
#         table = "<table class='table'>"
#         table += "<thread><tr><th>" + "</th><th>".join(headers) + "</th></tr></thead>"
#         table += "<tbody>"
#
#         current_sobeys_category = None
#         current_walmart_category = None
#         for r in sorted_list:
#             if current_sobeys_category != r["Sobeys_category"]:
#                 current_sobeys_category = r["Sobeys_category"]
#                 table += f"<tr><td colspan='4'><strong>{current_sobeys_category}</strong></td></tr>"
#
#             if current_walmart_category != r['Walmart_category']:
#                 current_walmart_category = r['Walmart_category']
#                 # Fill the first 4 columns if there's no change in Sobeys category to maintain the table structure
#                 if current_sobeys_category == r['Sobeys_category']:
#                     table += f"<tr><td colspan='4'></td><td colspan='4'><strong>{current_walmart_category}</strong></td></tr>"
#                 else:
#                     table += f"<tr><td colspan='4'><strong>{current_walmart_category}</strong></td></tr>"
#
#             # Add the data rows
#             table += "<tr>"
#             for header in headers:
#                 if not header.endswith("_category"):  # exclude category columns from normal rows
#                     table += f"<td>{r[header]}</td>"
#             table += "</tr>"
#
#         table += "</tbody></table>"
#         return Markup(table)

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

    print(f"done {filePath}")
    sobeys_table, walmart_table = read_csv(filePath)

    return jsonify(sobeys=sobeys_table, walmart=walmart_table)

@app.route("/recommendations",methods=['GET'])
def showRecommendations():
    pass

def main():
    # urls = ["https://www.sobeys.com/en/flyer/", "https://www.walmart.ca/en/flyer"]
    # for url in urls:
    #     sobeys_walmart_flyer(url)
    showFlyers()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # main()

