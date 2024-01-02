# Name: Dong Han
# Mail: dongh@mun.ca

# Used to clean the saved csv to make it more struct

import pandas as pd
import re,os

def parse_aria_label(aria_label):
    # Extracting name, which is before the first comma
    name = aria_label.split(',')[0].strip()

    # Extracting price, which is any dollar amount in the string
    price_matches = re.findall(r'\$\d+(?:\.\d+)?(?: /lb)?', aria_label)
    price = price_matches[0] if price_matches else ''


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
    df = pd.read_csv(fileName)
    parsed_aria_label_combined_rest = [(*parse_aria_label(label),item_type,item_id,item_type_number)
                         for label,item_type,item_id,item_type_number in
                         zip(df["aria-label"],df["item-type"],df["item-id"],df["item-type-number"])
                         if parse_aria_label(label)[1]] # to check if price is none
    new_df = pd.DataFrame(parsed_aria_label_combined_rest,
                             columns=["name", "price", "measurement", "remark","item-type","item-id","item-type-number"])

    newFileName = "cleaned_"+os.path.basename(fileName)

    folder_path = 'cleaned_data'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    new_file_path = os.path.join(folder_path, newFileName)

    new_df.to_csv(new_file_path,index=False)
    print(f"the {shopName} data is Cleaned and Saved as '{newFileName}' in folder '{folder_path}'")


def main():
    clean_data(r"scraped_draft_data/sobeysFlyer_2024-01-02.csv","sobeys")

if __name__ == '__main__':
    main()
