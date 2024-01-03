# Used to check groceries' flyers

from flask import Flask,request,render_template,jsonify,Response
import getFlyer_sobeys_walmart,data_clean
import os
from datetime import datetime

app = Flask(__name__,static_folder='assets')

def sobeys_walmart_flyer(url):
    draft_flyer_file,shopName = getFlyer_sobeys_walmart.getFlyer(url)
    if draft_flyer_file == -1:
        print("error in getting flyer data")
        return
    cleaned_filePath = data_clean.clean_data(draft_flyer_file,shopName)
    return cleaned_filePath

def read_csv(filePath):
    with open(filePath, 'r', newline='',encoding="utf-8") as file:
        return file.read()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/flyers",methods=['GET'])
def showFlyers():
    folder_path = 'combined_data'
    fileName = "combined_" + datetime.now().strftime("%Y-%m-%d") + ".csv"
    filePath = os.path.join(folder_path, fileName)
    if os.path.exists(filePath):
        print(f"done {filePath}")
        content = read_csv(filePath)
        return Response(content, mimetype='text/csv', headers={"Content-disposition": "attachment; filename=" + fileName})

    cleanedFilesPaths = []
    urls = ["https://www.sobeys.com/en/flyer/", "https://www.walmart.ca/en/flyer"]
    for url in urls:
        cleaned_filePath = sobeys_walmart_flyer(url)
        cleanedFilesPaths.append(cleaned_filePath)

    data_clean.combine_data(cleanedFilesPaths[0],cleanedFilesPaths[1])

    print(f"done {filePath}")
    content = read_csv(filePath)
    return Response(content, mimetype='text/csv', headers={"Content-disposition": "attachment; filename=" + fileName})



def main():
    # urls = ["https://www.sobeys.com/en/flyer/", "https://www.walmart.ca/en/flyer"]
    # for url in urls:
    #     sobeys_walmart_flyer(url)
    showFlyers()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # main()

