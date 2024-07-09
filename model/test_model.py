import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import  DataLoader
from torch.optim import AdamW
from datetime import datetime
from typing import List
import csv

from sklearn.model_selection import train_test_split
import numpy as np
import wandb
import sys

# make it possible to find the data python module in parent folder
sys.path.insert(0, '..') 

# self written classes/modules
import dataset 
from evaluation.test_metrics import test_model_performance, plot_confusion_matrix
from evaluation import ece_loss
from model.load_pretrained_model import load_pretrained_model
from util_model import *


NUM_CLASSES = 2
CASE_TESTSET = ['hw','mg_llama','mg_mixed','mg_gpt_special','mg_mistral_special']#,'mg_gpt_special','mg_mistral_special']  # ['hw','mg_llama','mg_mixed','mg_gpt_special','mg_mistral_special']# 
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

#wandb = initialize_wandb()
#timestamp = datetime.now().strftime("%m%d-%H%M")
run = wandb.init(project="Test Datasets Calibration")


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
device = torch.device("cuda:0")#"cuda" if torch.cuda.is_available() else "cpu")
#val_dataset = Dataset.from_dict({"text": texts, "label": labels}).with_format("torch", device = device)
val_dataset = dataset.TextDataset(texts, labels, tokenizer, max_length=512)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
#print(val_loader)
# Define parameters
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
ece_criterion = ece_loss._ECELoss().cuda()
    
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

        # get the loss in validation
        loss = criterion(outputs.logits, labels_from_samples)
        total_loss += loss.item()
        total_samples += len(labels_from_samples)


logits = torch.cat(logits_list).cuda()
labels = torch.cat(labels_list).cuda()
ece = ece_criterion(logits, labels).item()

print("ECE",ece)
# calculate loss validation to estimate how model generalizes to unseen data
val_loss = total_loss / len(val_loader)

metrics_dict = test_model_performance(model_name, 
                                    all_val_labels, 
                                    all_val_preds, 
                                    all_val_probs, 
                                    val_loss=val_loss,
                                    ece_value=ece,
                                                )
conf_matrix_digits = metrics_dict["conf_matrix"]
conf_matrix_image = plot_confusion_matrix(conf_matrix_digits)

metrics_dict.update({
        "Loss/Validation": val_loss,
        "conf_matrix": wandb.Image(conf_matrix_image),
    })



wandb.log(metrics_dict)
wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                                                    y_true=all_val_labels, 
                                                    preds=all_val_preds,
                                                    class_names=[0,1])})

# end of test run
wandb.finish()