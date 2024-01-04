# Name: Dong Han
# Mail: dongh@mun.ca

# Used to clean the saved csv to make it more struct

import pandas as pd
import re,os
from datetime import datetime
from grocery_model import use_model

def parse_aria_label(aria_label):
    # Extracting name, which is before the first comma
    name = aria_label.split(',')[0].strip()

    # Extracting price, which is any dollar amount in the string
    price_matches = re.findall(r'\$\d+(?:\.\d+)?(?: /lb)?', aria_label)
    price = price_matches[0] if price_matches and "$" not in name else ''
    if "Save" in aria_label and len(price_matches) > 1:
        price = price_matches[1]
    # print(price,name)

    # Extracting measurement, which could be 'lb' or a number like '3/' indicating quantity
    measurement = '1'  # default value
    if '/lb' in price:
        measurement = 'lb'
    else:
        quantity_matches = re.findall(r'(\d+)/ \$', aria_label)
        if quantity_matches:
            measurement = quantity_matches[0]

    price = re.sub(r' /lb', '', price) # remove '/lb' from price

    # Remark is the entire aria-label
    remark = aria_label.strip()

    return name, price, measurement, remark

def clean_data(fileName,shopName):
    newFileName = "cleaned_"+os.path.basename(fileName)

    folder_path = 'cleaned_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Check if the cleaned data already there
    new_file_path = os.path.join(folder_path, newFileName)
    if os.path.exists(new_file_path):
        return new_file_path

    # If not exists, continue the clean process
    df = pd.read_csv(fileName)
    parsed_aria_label_combined_rest = [(*parse_aria_label(label),item_type,item_id,item_type_number)
                         for label,item_type,item_id,item_type_number in
                         zip(df["aria-label"],df["item-type"],df["item-id"],df["item-type-number"])
                         if parse_aria_label(label)[1]] # to check if price is none
    new_df = pd.DataFrame(parsed_aria_label_combined_rest,
                             columns=["Item_Name", "Price", "UoM", "remark","item-type","item-id","item-type-number"])

    new_df = new_df.drop_duplicates(subset=["Item_Name","Price"])

    new_df.to_csv(new_file_path,index=False)

    use_model.classify_grocery_data(new_file_path) # add a category n save back

    print(f"the {shopName} data is Cleaned and Saved as '{newFileName}' in folder '{folder_path}'")
    return new_file_path

def combine_data(file1,file2):
    folder_path = 'combined_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    today_date = datetime.now().strftime("%Y-%m-%d")
    fileName = "combined_"+today_date+".csv"
    filePath = os.path.join(folder_path,fileName)

    if os.path.exists(filePath):
        return filePath

    df1 = pd.read_csv(file1,usecols=[0,1,2,8])
    df2 = pd.read_csv(file2,usecols=[0,1,2,8])

    # Add a header row for labeling
    df1.columns = ["Sobeys_" + str(col) for col in df1.columns]
    df2.columns = ["Walmart_" + str(col) for col in df2.columns]

    combined_df = pd.concat([df1,df2],axis=1)

    combined_df.to_csv(filePath, index=False)
    print(f"the Combined Data is Saved as '{fileName}' in folder '{folder_path}'")

    return filePath

def main():
    # clean_data(r"scraped_draft_data/walmartFlyer_2024-01-03.csv","sobeys")
    # dataFile1 = "cleaned_data/cleaned_sobeysFlyer_2024-01-02.csv"
    # dataFile2 = "cleaned_data/cleaned_walmartFlyer_2024-01-02.csv"
    #
    # f = combine_data(dataFile1, dataFile2)
    # print(f)

    t = "Save $60, ,"
    parse_aria_label(t)

if __name__ == '__main__':
    main()
