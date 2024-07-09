import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from typing import List
import csv
import numpy as np
import sys

# make it possible to find the data python module in parent folder
sys.path.insert(0, '..') 

# self written classes/modules
import model.dataset 
from model.load_pretrained_model import load_pretrained_model
from model.util_model import *


NUM_CLASSES = 2
CASE_TESTSET = ['mg_gpt_special','mg_mistral_special','hw','mg_llama','mg_mixed']  # ['hw','mg_llama','mg_mixed',]# 
CASE_MODEL = 'llama' #['llama','mixed']
testsets_hw = [FILENAME_HW_DATASET_TEXTS_TEST]
testsets_mg = [FILENAME_MG_DATASET_TEXTS_TEST_LLAMA,
            FILENAME_MG_DATASET_TEXTS_TEST_MIXED,
            FILENAME_MG_DATASET_TEXTS_TEST_SPECIAL_GPT,
            FILENAME_MG_DATASET_TEXTS_TEST_SPECIAL_MISTRAL 
            ]
def load_dataset(FILENAME) -> List[str]:
    texts = []
    os_path_object = os.path.join(OS_PATH_DATASET_TEXTS_ONLY, FILENAME)
    with open(os_path_object, 'r', encoding='utf-8') as file:
        for line in file:
            texts.append(line.strip())
    return texts

def load_dataset_tsv(FILENAME) -> List[str]:
    first_column = []
    os_path_object = os.path.join(OS_PATH_DATASET_TEXTS_ONLY_TEST_SPECIAL, FILENAME)
    with open(os_path_object, 'r', newline='', encoding='utf-8') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            if row:  # Check if the row is not empty
                first_column.append(row[0])
                #print(row[0][:19])
    return first_column

# Load dataset and split it into training and validation sets
case_testset = CASE_TESTSET
texts = []
labels = []
if 'hw' in case_testset :
    temp_text = load_dataset(FILENAME_HW_DATASET_TEXTS_TEST)
    texts.extend(temp_text)
    labels.extend([0]* len(temp_text))
if 'mg_llama' in case_testset:
    temp_text = load_dataset(FILENAME_MG_DATASET_TEXTS_TEST_LLAMA)
    texts.extend(temp_text)
    labels.extend([1]* len(temp_text))
if 'mg_mixed' in case_testset:
    temp_text = load_dataset(FILENAME_MG_DATASET_TEXTS_TEST_MIXED)
    texts.extend(temp_text)
    labels.extend([1]* len(temp_text))
if 'mg_gpt_special' in case_testset:
    temp_text = load_dataset_tsv(FILENAME_MG_DATASET_TEXTS_TEST_SPECIAL_GPT)
    texts.extend(temp_text)
    labels.extend([1]* len(temp_text))
if 'mg_mistral_special' in case_testset:
    temp_text = load_dataset_tsv(FILENAME_MG_DATASET_TEXTS_TEST_SPECIAL_MISTRAL)
    texts.extend(temp_text)
    labels.extend([1]* len(temp_text))


tokenizer = None
model = None
model_name = None
case_model = CASE_MODEL
if case_model == 'llama':
    tokenizer, model, model_name = load_pretrained_model('llama')
if case_model == 'mixed':
    tokenizer, model, model_name = load_pretrained_model('mixed')


# Create datasets and data loaders
device = torch.device("cuda:0")#"cuda" if torch.cuda.is_available() else "cpu")
val_dataset = model.dataset.TextDataset(texts, labels, tokenizer, max_length=512)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
#print(val_loader)
# Define parameters
model.to(device)
criterion = torch.nn.CrossEntropyLoss()

    
############ Validation ###########
model.eval()

# Initialize as empty NumPy arrays
all_val_preds = np.array([], dtype=int)
all_val_labels = np.array([], dtype=int)
all_val_probs = np.empty((0, NUM_CLASSES), dtype=float)

total_loss = 0.0
total_samples = 0
logits_list = []
labels_list = []
data_tp = []
data_tn = []
data_fp = []
data_fn = []
data_for_error_analysis = []
counter_tp = 0
counter_tn = 0
counter_fp = 0
counter_fn = 0

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
        predicted_labels_to_validate = torch.argmax(logits, dim=1)

        # .cpu(): moves the tensor from any available device (e.g., GPU) to the CPU.
        # If the tensor is already on the CPU, it has no effect.
        # .numpy(): converts the tensor to a NumPy array.
        labels_to_add = labels_from_samples.cpu().numpy()
        predictions_to_add = predicted_labels_to_validate.cpu().numpy()
        probabilities_to_add = probabilities_to_validate.cpu().numpy()

        # Concatenate the arrays
        logits_list.append(logits)
        labels_list.append(labels_from_samples)
        
        all_val_labels = np.concatenate([all_val_labels, labels_to_add])
        all_val_preds = np.concatenate([all_val_preds, predictions_to_add])
        all_val_probs = np.concatenate([all_val_probs, probabilities_to_add])

        decoded_text_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        cls_index = decoded_text_tokens.index('[CLS]')

        # Find the index of [PAD] token
        try:
            pad_index = decoded_text_tokens.index('[PAD]')
        except:
            pad_index = len(decoded_text_tokens) - 1
        # Extract the text between [CLS] and [PAD]
        extracted_tokens = decoded_text_tokens[cls_index+1:pad_index]
        #print(extracted_t)
        # Convert tokens to text
        text = tokenizer.convert_tokens_to_string(extracted_tokens)
        associated_label = predictions_to_add[0]
        correct_label = labels_to_add[0]

        if associated_label == correct_label ==1:
            if counter_tp >= 100:
                continue
            counter_tp +=1

            data_tp.append({
                "text": text,
                "predicted_label": associated_label,
                "correct_label": correct_label
            })
        if associated_label == correct_label ==0:
            if counter_tn >= 100:
                continue
            counter_tn +=1

            data_tn.append({
                "text": text,
                "predicted_label": associated_label,
                "correct_label": correct_label
            })
        if not associated_label == correct_label and correct_label == 1:
            if counter_fp >= 100:
                continue
            counter_fp +=1

            data_fp.append({
                "text": text,
                "predicted_label": associated_label,
                "correct_label": correct_label
            })
        if not associated_label == correct_label and correct_label == 0:
            if counter_fn >= 100:
                continue
            counter_fn +=1

            data_fn.append({
                "text": text,
                "predicted_label": associated_label,
                "correct_label": correct_label
            })
        if counter_fn == counter_fp == counter_tn == counter_tp == 100:
            break



with open('dataset_for_error_analysis.tsv', 'w') as f:
    f.write("Text\tPredicted Label\tCorrect Label\n")
    for entry in data_tp:
        line = entry["text"] + '\t' + str(entry["predicted_label"]) + '\t' + str(entry["correct_label"]) + '\n'
        f.write(line)
    for entry in data_tn:
        line = entry["text"] + '\t' + str(entry["predicted_label"]) + '\t' + str(entry["correct_label"]) + '\n'
        f.write(line)
    for entry in data_fp:
        line = entry["text"] + '\t' + str(entry["predicted_label"]) + '\t' + str(entry["correct_label"]) + '\n'
        f.write(line)
    for entry in data_fn:
        line = entry["text"] + '\t' + str(entry["predicted_label"]) + '\t' + str(entry["correct_label"]) + '\n'
        f.write(line)
