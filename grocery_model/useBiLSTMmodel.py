# Name: Dong Han
# Mail: dongh@mun.ca

# Used to implement trained BiLSTMWithXLMRModel to predict the category

import torch
from grocery_model import BiLSTMWithXLMRModel
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import pickle


def ini_model():
    n_cate = 11
    n_subcate = 90
    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    base_model = XLMRobertaModel.from_pretrained(model_name)

    binary_mask_path = "binary_mask_tensor.pt"

    # Load the saved binary mask tensor
    binary_mask_tensor = torch.load(binary_mask_path, map_location=torch.device('cpu'))

    model = BiLSTMWithXLMRModel.BiLSTMWithXLMRModel(base_model, n_cate, n_subcate,binary_mask_tensor)
    model_path = "model_epoch_5.pt"
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

    # input_text = "ORGANIC Smooth Peanut Butter"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    # Perform a prediction
    with torch.no_grad():  # This tells PyTorch that we do not need to compute gradients during inference
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        category_probs, subcategory_probs = model(input_ids=input_ids, attention_mask=attention_mask)

        #for debug
        print("category_probs: ",category_probs)
        print("subcategory_probs: ", subcategory_probs)

    # Process your model's outputs as needed (e.g., applying argmax to get the most likely category)
    predicted_category = torch.argmax(category_probs, dim=1)
    predicted_subcategory = torch.argmax(subcategory_probs, dim=1)

    # Convert to numpy if they're in torch tensor format
    predicted_category = predicted_category.numpy()
    predicted_subcategory = predicted_subcategory.numpy()

    # Convert numerical labels to human-readable labels
    human_readable_categories = [category_label_mapping[label] for label in predicted_category]
    human_readable_subcategories = [subcategory_label_mapping[label] for label in predicted_subcategory]

    print("Predicted Categories:", human_readable_categories)
    print("Predicted Subcategories:", human_readable_subcategories)

def main():
    model,tokenizer = ini_model()
    category_label_mapping = load_category_mapping("category_label_mapping.pkl")
    subcategory_label_mapping = load_category_mapping("subcategory_label_mapping.pkl")

    input_texts = ["Fresh Pork Loin Back Ribs","ORGANIC Smooth Peanut Butter", "Kombucha or Sparkling Drink",
                   "ASIAN INSPIRATIONS, WONG WING Frozen Entr??es, Egg Rolls, Spring Rolls or Dumplings, , $5 EACH WHEN YOU BUY 2 OR MORE."]
    input_text1 = ["Water Bottle - Orange","Cereal Flip Lid Container/Storage Jar - Assorted Colour","Fresh Catch Fish Fingers","Pork Classic Salami"]
    for input_text in input_text1:
        model_prediction(model, tokenizer, category_label_mapping, subcategory_label_mapping,input_text)


if __name__ == '__main__':
    main()

