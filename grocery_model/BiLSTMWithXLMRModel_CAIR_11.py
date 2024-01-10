# Name: Dong Han
# Mail: dongh@mun.ca

import os,sys
import traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import pandas as pd
import pickle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# CAIR Conda environment folder
specific_folder_path = r'/gpfs/home/dongh/miniconda3/envs/stylegan3/lib'
sys.path.append(specific_folder_path)

# Used for creating log file
current_file_name = os.path.basename(__file__)

model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
base_model = XLMRobertaModel.from_pretrained(model_name)

# Helper Func to Print n Save to file
def log_print(*args, **kwargs):
    with open(current_file_name+'_CAIR_output_log11.txt', 'a') as log_file:
        print(*args, **kwargs, file=log_file)  # Save to file
    print(*args, **kwargs)  # Print to console

log_print("after importing...")

def preprocess_csvdata(grocery_dataset):
    df = pd.read_csv(grocery_dataset)
    df = df.dropna(subset=['product'])

    # Tokenize 'product' column
    df['tokenized_product'] = df['product'].apply(
        lambda x: tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )['input_ids'].squeeze(0)
    )
    # Convert 'category' and 'sub_category' to numerical labels
    category_labels = {category: idx for idx, category in enumerate(df['category'].unique())}
    subcategory_labels = {subcategory: idx for idx, subcategory in enumerate(df['sub_category'].unique())}

    df['category_label'] = df['category'].map(category_labels)
    df['sub_category_label'] = df['sub_category'].map(subcategory_labels)

    # Inverse mappings and save them
    category_label_mapping = {idx: category for category, idx in category_labels.items()}
    subcategory_label_mapping = {idx: subcategory for subcategory, idx in subcategory_labels.items()}

    with open('category_label_mapping11.pkl', 'wb') as f:
        pickle.dump(category_label_mapping, f)

    with open('subcategory_label_mapping11.pkl', 'wb') as f:
        pickle.dump(subcategory_label_mapping, f)

    # Initialize binary mask matrix
    binary_mask = np.zeros((len(df['category'].unique()), len(df['sub_category'].unique())), dtype=np.float32)
    # Fill the mask matrix
    for _, row in df.iterrows():
        category_idx = category_labels[row['category']]
        subcategory_idx = subcategory_labels[row['sub_category']]
        binary_mask[category_idx, subcategory_idx] = 1

    binary_mask_tensor = torch.tensor(binary_mask)

    # After creating the binary mask tensor in training
    torch.save(binary_mask_tensor, 'binary_mask_tensor11.pt')

    log_print("done preprocess data")
    return df,binary_mask_tensor

class ProductDataset(Dataset):
    def __init__(self,dataframe,max_token_len=256):
        self.dataframe = dataframe
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data_row = self.dataframe.iloc[idx]
        product_text = data_row.tokenized_product

        category_label = torch.tensor(data_row.category_label)
        subcategory_label = torch.tensor(data_row.sub_category_label)

        return {
            'input_ids': product_text,
            'category_label': category_label,
            'subcategory_label': subcategory_label
        }

class BiLSTMWithXLMRModel(nn.Module):
    def __init__(self, base_model, n_cate, n_subcate,binary_mask):
        super().__init__()
        self.base_model = base_model
        self.caetegoryClassifier = nn.Linear(base_model.config.hidden_size, n_cate)
        self.subcaetegoryClassifier = nn.Linear(base_model.config.hidden_size, n_subcate)

        self.dropout = nn.Dropout(p=0.2)  # p=0.2 for a mild regularization
        self.binary_mask = binary_mask

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

def collate_batch(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True,padding_value=tokenizer.pad_token_id)
    category_labels = torch.tensor([item['category_label'] for item in batch],dtype=torch.long)
    subcategory_labels = torch.tensor([item['subcategory_label'] for item in batch],dtype=torch.long)
    attention_masks = (input_ids != tokenizer.pad_token_id).type(torch.FloatTensor)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'category_label': category_labels,
        'subcategory_label': subcategory_labels
    }

######################################################### MAIN TRAIN ##################################################
def main():
    try:


        # Directory where the model files will be saved
        model_save_dir = "saved_models"
        os.makedirs(model_save_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_print(f"Using device: {device}")

        grocery_dataset = "BigBasket_Products.csv"
        df,binary_mask_tensor = preprocess_csvdata(grocery_dataset)
        dataset = ProductDataset(df)

        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_dataset = ProductDataset(train_df)
        val_dataset = ProductDataset(val_df)

        # Create DataLoader for both sets
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

        # Generate model
        n_cate = 11
        n_subcate = 90
        binary_mask_tensor = binary_mask_tensor.to(device)
        model = BiLSTMWithXLMRModel(base_model,n_cate,n_subcate,binary_mask_tensor).to(device)

        # Compute class weights for category
        category_class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(df['category_label']),
            y=df['category_label'].values
        )
        category_class_weights_tensor = torch.tensor(category_class_weights, dtype=torch.float, device=device)

        # Compute class weights for subcategory
        subcategory_class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(df['sub_category_label']),
            y=df['sub_category_label'].values
        )
        subcategory_class_weights_tensor = torch.tensor(subcategory_class_weights, dtype=torch.float, device=device)

        # Initialize the loss function with class weights
        # loss_func = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

        optimizer = torch.optim.Adam(model.parameters(),lr=0.00003)

        log_print("done df, model")

        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                # Unpack the batch
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
                category_labels = batch['category_label'].to(device)
                subcategory_labels = batch['subcategory_label'].to(device)

                # Forward pass: Ensure model accepts these as named arguments
                category_probs, subcategory_probs = model(input_ids=input_ids, attention_mask=attention_mask)

                # Compute loss
                category_loss = torch.nn.CrossEntropyLoss(weight=category_class_weights_tensor)(category_probs,
                                                                                                category_labels)
                subcategory_loss = torch.nn.CrossEntropyLoss(weight=subcategory_class_weights_tensor)(subcategory_probs,
                                                                                                      subcategory_labels)
                total_loss = category_loss + subcategory_loss
                train_loss += total_loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            avg_train_loss = train_loss / len(train_loader)

            # Val phase
            model.eval()
            val_loss = 0
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
                category_labels = batch['category_label'].to(device)
                subcategory_labels = batch['subcategory_label'].to(device)

                category_probs, subcategory_probs = model(input_ids=input_ids, attention_mask=attention_mask)

                category_loss = torch.nn.CrossEntropyLoss(weight=category_class_weights_tensor)(category_probs,
                                                                                                category_labels)
                subcategory_loss = torch.nn.CrossEntropyLoss(weight=subcategory_class_weights_tensor)(subcategory_probs,
                                                                                                      subcategory_labels)
                total_loss = category_loss + subcategory_loss

                val_loss += total_loss.item()
            avg_val_loss = val_loss / len(val_loader)

            log_print(f"Epoch {epoch}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")

            model_save_path = os.path.join(model_save_dir,f"model_epoch_{epoch}_11.pt")
            torch.save(model.state_dict(),model_save_path)
            log_print(f"Saved model to {model_save_path}")

    except Exception as e:
        # Save the error output to a text file
        output_file = f"/gpfs/home/dongh/groceryClassify/{current_file_name}_error_log11.txt"
        with open(output_file, "a") as file:
            file.write(f"An error occurred: {e}\n")
            file.write("Traceback (most recent call last):\n")
            traceback.print_exc(file=file)

if __name__ == '__main__':
    main()


