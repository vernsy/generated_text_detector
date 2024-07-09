from typing import Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import sys
sys.path.insert(0, '..') 
from setup import config

def load_pretrained_model(model_name: str) -> Union[AutoTokenizer, AutoModelForSequenceClassification,str]:
    
    if model_name == 'distilbert':
        # Initialize the tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    elif model_name == 'bert':
        # Initialize the tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 'bert-large-uncased', 'bert-base-cased', etc.
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    elif model_name == 'roberta':
        # Initialize the tokenizer and model
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # 'bert-large-uncased', 'bert-base-cased', etc.
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    elif model_name == 'llama':
        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.OS_PATH_TO_FINETUNED_MODEL_LLAMA)  # 'bert-large-uncased', 'bert-base-cased', etc.
        model = AutoModelForSequenceClassification.from_pretrained(config.OS_PATH_TO_FINETUNED_MODEL_LLAMA)
    elif model_name == 'mixed':
        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.OS_PATH_TO_FINETUNED_MODEL_MIXED)  # 'bert-large-uncased', 'bert-base-cased', etc.
        model = AutoModelForSequenceClassification.from_pretrained(config.OS_PATH_TO_FINETUNED_MODEL_MIXED)
    else:
        raise ValueError("Invalid input: No matching string for model_name found!.")


    return tokenizer, model, model_name