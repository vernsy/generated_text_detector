import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import  DataLoader
from typing import List
import argparse

import numpy as np
import sys

# make it possible to find the data python module in parent folder
sys.path.insert(0, '..') 

# self written classes/modules
import dataset 
from model.load_pretrained_model import load_pretrained_model
from util_model import *

CASE_MODEL = 'mixed'

def detector(input_text:str) -> float:
    texts = [input_text]
    labels = [0] # does not matter which label

    #print(labels)

    tokenizer = None
    model = None
    model_name = None
    case_model = CASE_MODEL
    if case_model == 'llama':
        tokenizer, model, model_name = load_pretrained_model('llama')
    if case_model == 'mixed':
        tokenizer, model, model_name = load_pretrained_model('mixed')


    # Create datasets and data loaders
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
    #val_dataset = dataset.TextDataset(texts, labels, tokenizer, max_length=512)

    encoding = tokenizer(input_text, truncation=True, padding='max_length',
                            return_tensors='pt')

    batch_dict =  [{
        'input_ids': encoding['input_ids'].squeeze().to('cpu'),
        'attention_mask': encoding['attention_mask'].squeeze().to('cpu'),
        'label': torch.tensor(labels[0], dtype=torch.long).to('cpu')
    }]

    val_loader = DataLoader(batch_dict, batch_size=1, shuffle=False)
    model.to(device)
        
    ############ Validation ###########
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels_from_samples = batch['label']

            outputs = model(input_ids, attention_mask=attention_mask)

            # Raw and unnormalized predictions
            logits = outputs.logits

            # Apply softmax to get probabilities
            probabilities_to_validate = F.softmax(logits, dim=1)

            # Get the predicted class labels
            predicted_label = torch.argmax(logits, dim=1)
            probability = probabilities_to_validate[:,1].numpy()[0]
            probability_rounded = np.round(probability*100,2)
            label = predicted_label.numpy()[0]

            return probability_rounded, label
        
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Add the text you want to test')

    # Add arguments
    parser.add_argument('input_text', type=str, help='Your text')

    # Parse arguments
    args = parser.parse_args()

    # Access the parsed arguments
    input_text = args.input_text

    # Your main logic here
    print(detector(input_text))

if __name__ == '__main__':
    main()

