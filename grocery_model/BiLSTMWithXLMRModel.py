# Name: Dong Han
# Mail: dongh@mun.ca

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import pandas as pd


model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
base_model = XLMRobertaModel.from_pretrained(model_name)

def preprocess_csvdata(grocery_dataset):
    df = pd.read_csv(grocery_dataset)
    df = df.dropna(subset=['product'])

    # Tokenize 'product' column
    df['tokenized_product'] = df['product'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    # Convert 'category' and 'sub_category' to numerical labels
    category_labels = {category: idx for idx, category in enumerate(df['category'].unique())}
    subcategory_labels = {subcategory: idx for idx, subcategory in enumerate(df['sub_category'].unique())}

    df['category_label'] = df['category'].map(category_labels)
    df['sub_category_label'] = df['sub_category'].map(subcategory_labels)

    return df

class ProductDataset(Dataset):
    def __init__(self,dataframe,max_token_len=256):
        self.dataframe = dataframe
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data_row = self.dataframe.iloc[idx]
        product_text = data_row.tokenized_product

        product_text = torch.tensor(
            product_text[:self.max_token_len] + [0] * (self.max_token_len - len(product_text[:self.max_token_len])))

        category_label = torch.tensor(data_row.category_label)
        subcategory_label = torch.tensor(data_row.sub_category_label)

        return {
            'input_ids': product_text,
            'category_label': category_label,
            'subcategory_label': subcategory_label
        }

class BiLSTMWithXLMRModel(nn.Module):
    def __init__(self, base_model, n_cate, n_subcate):
        super().__init__()
        self.base_model = base_model
        self.caetegoryClassifier = nn.Linear(base_model.config.hidden_size, n_cate)
        self.subcaetegoryClassifier = nn.Linear(base_model.config.hidden_size, n_subcate)

        # Example binary mask; in practice, you'd create this based on your data
        self.binary_mask = torch.rand(n_cate, n_subcate) > 0.5

    def forward(self,input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids,attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:,0,:]

        category_logits = self.caetegoryClassifier(sequence_output)
        subcategory_logits = self.subcaetegoryClassifier(sequence_output)

        # Apply Dynamic Masked Softmax
        softmax = nn.Softmax(dim=1)
        categoty_probs = softmax(category_logits)
        selected_category = torch.argmax(categoty_probs,dim=-1)
        mask = self.binary_mask[selected_category]
        masked_subcateroy_logits = subcategory_logits * mask.float()
        subcategoty_probs = softmax(masked_subcateroy_logits)

        return categoty_probs,subcategoty_probs

######################################################### MAIN TRAIN ##################################################
def main():

    # Directory where the model files will be saved
    model_save_dir = "saved_models"
    os.makedirs(model_save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    grocery_dataset = "BigBasket_Products.csv"
    df = preprocess_csvdata(grocery_dataset)
    dataset = ProductDataset(df)

    # Create DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Generate model
    n_cate = 11
    n_subcate = 90
    model = BiLSTMWithXLMRModel(base_model,n_cate,n_subcate).to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    print("done df, model")

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            # Unpack the batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
            category_labels = batch['category_label'].to(device)
            subcategory_labels = batch['subcategory_label'].to(device)

            # Forward pass: Ensure model accepts these as named arguments
            category_probs, subcategory_probs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute loss
            category_loss = loss_func(category_probs, category_labels)
            subcategory_loss = loss_func(subcategory_probs, subcategory_labels)
            total_loss = category_loss + subcategory_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {total_loss.item()}")

        model_save_path = os.path.join(model_save_dir,f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(),model_save_path)
        print(f"Saved model to {model_save_path}")


if __name__ == '__main__':
    main()

