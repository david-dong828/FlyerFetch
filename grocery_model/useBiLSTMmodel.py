# Name: Dong Han
# Mail: dongh@mun.ca

# Used to implement trained BiLSTMWithXLMRModel to predict the category

import torch
from grocery_model import BiLSTMWithXLMRModel
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import pickle
import os
from datetime import datetime
import pandas as pd


def ini_model():
    n_cate = 11
    n_subcate = 90
    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    base_model = XLMRobertaModel.from_pretrained(model_name)

    binary_mask_path = "grocery_model/binary_mask_tensor11.pt"

    # Load the saved binary mask tensor
    binary_mask_tensor = torch.load(binary_mask_path, map_location=torch.device('cpu'))

    model = BiLSTMWithXLMRModel.BiLSTMWithXLMRModel(base_model, n_cate, n_subcate,binary_mask_tensor)
    model_path = "grocery_model/model_epoch_13_11.pt"
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # print(state_dict.keys())

    # Remove the unexpected key- base_model.embeddings.position_ids
    state_dict.pop("base_model.embeddings.position_ids", None)

    model.load_state_dict(state_dict)

    return model,tokenizer

def load_category_mapping(map_path):
    with open(map_path, 'rb') as f:
        map_path = pickle.load(f)
    return map_path

def model_prediction(model,tokenizer,category_label_mapping,subcategory_label_mapping,input_text):
    model.eval() # set to evaluation mode

    if isinstance(input_text,pd.Series):
        input_text = input_text.tolist()

    # input_text = "ORGANIC Smooth Peanut Butter"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    # Perform a prediction
    with torch.no_grad():  # Tells PyTorch that we do not need to compute gradients during inference
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        category_probs, subcategory_probs = model(input_ids=input_ids, attention_mask=attention_mask)

        #for debug
        # print("category_probs: ",category_probs)
        # print("subcategory_probs: ", subcategory_probs)

    # Process your model's outputs as needed (e.g., applying argmax to get the most likely category)
    predicted_category = torch.argmax(category_probs, dim=1)
    predicted_subcategory = torch.argmax(subcategory_probs, dim=1)

    # Convert to numpy if they're in torch tensor format
    # convert To list for integrating back to Dataframe
    predicted_category = predicted_category.numpy().tolist()
    predicted_subcategory = predicted_subcategory.numpy().tolist()

    # Convert numerical labels to human-readable labels
    human_readable_categories = [category_label_mapping[label] for label in predicted_category]
    human_readable_subcategories = [subcategory_label_mapping[label] for label in predicted_subcategory]

    # print("Predicted Categories:", human_readable_categories)
    # print("Predicted Subcategories:", human_readable_subcategories)

    return human_readable_categories,human_readable_subcategories

def add_predicted_categories(for_predict_file):
    model, tokenizer = ini_model()
    category_label_mapping = load_category_mapping("grocery_model/category_label_mapping11.pkl")
    subcategory_label_mapping = load_category_mapping("grocery_model/subcategory_label_mapping11.pkl")

    df = pd.read_csv(for_predict_file)

    predicated_categories, predicated_subcategories = model_prediction(model, tokenizer, category_label_mapping,
                                                                       subcategory_label_mapping, df["remark"])
    df["predict_category"] = predicated_categories
    df["predict_subcategory"] = predicated_subcategories

    df.to_csv(for_predict_file, index=False)  # save back
    print("Added predicted categories successfully to Clean Data!")

    folder_path = 'recommendation'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    today_date = datetime.now().strftime("%Y-%m-%d")
    if "sobeys" in for_predict_file:
        fileName = "sobeys_" + "addedCategory_" + today_date + ".csv"
    else:
        fileName = "walmart_" + "addedCategory_" + today_date + ".csv"
    filePath = os.path.join(folder_path, fileName)

    df.to_csv(filePath, index=False)
    print("Added predicted categories successfully to Recommendation!")

def main():
    for_predict_file = "../cleaned_data/cleaned_sobeysFlyer_2024-01-02.csv"
    add_predicted_categories(for_predict_file)


    # input_texts = ["Fresh Pork Loin Back Ribs","ORGANIC Smooth Peanut Butter", "Kombucha or Sparkling Drink",
    #                "ASIAN INSPIRATIONS, WONG WING Frozen Entr??es, Egg Rolls, Spring Rolls or Dumplings, , $5 EACH WHEN YOU BUY 2 OR MORE."]
    # input_text1 = ["Water Bottle - Orange","Cereal Flip Lid Container/Storage Jar - Assorted Colour","Fresh Catch Fish Fingers","Pork Classic Salami"]
    # for input_text in input_texts:
    #     model_prediction(model, tokenizer, category_label_mapping, subcategory_label_mapping,input_text)


if __name__ == '__main__':
    main()

