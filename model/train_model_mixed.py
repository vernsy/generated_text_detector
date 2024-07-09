from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
                        TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy, \
                            DataCollatorWithPadding
import torch
import evaluate
import numpy as np
from datetime import datetime
import wandb
import sys
from typing import List
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# make it possible to find the data python module in parent folder
sys.path.insert(0, '..') 
# self written classes/modules
from evaluation.performance_metrics import compute_metrics_train_model
from util_model import *
# load dataset
def load_dataset(FILENAME) -> List[str]:
    texts = []
    os_path_object = os.path.join(OS_PATH_DATASET_TEXTS_ONLY, FILENAME)
    with open(os_path_object, 'r', encoding='utf-8') as file:
        for line in file:
            texts.append(line.strip())
    return texts




###########DATASET

# Load dataset and split it into training and validation sets
#def load_dataset_and_select_device():
hw_texts = load_dataset(FILENAME_HW_DATASET_TEXTS_TRAIN_EVAL)
hw_labels = [0]*len(hw_texts)
mg_texts_mixed = load_dataset(FILENAME_MG_DATASET_TEXTS_TRAIN_EVAL_MIXED)
mg_labels_mixed = [1]*len(mg_texts_mixed)

texts = hw_texts + mg_texts_mixed # gets later shuffled anyways
labels = hw_labels + mg_labels_mixed

# load on GPU -> do not allow cpu, because when GPU is full it tries CPU and then it can't find all tensors
device = torch.device("cuda:0")#"cuda" if torch.cuda.is_available() else "cpu")
ds = Dataset.from_dict({"text": texts, "label": labels}).with_format("torch", device = device)
print(ds[:3])
print(ds[22500:22503])
#return ds, device

    

###############EVALUATION
# Compute confusion matrix
def compute_confusion_matrix(true_labels, predicted_labels, labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    return cm

# Log confusion matrix to wandb
def log_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", 
                cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Log confusion matrix as wandb artifact
    artifact = wandb.Artifact("confusion_matrix", type="confusion_matrix")
    plt.savefig("confusion_matrix.png")
    artifact.add_file("confusion_matrix.png")
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
    #wandb.log_artifact(artifact)
    #wandb.log_confusion_matrix(confusion_matrix)

# Metric helper method
def compute_metrics(eval_pred):
    predicted_scores, labels = eval_pred
    predicted_labels = np.argmax(predicted_scores, axis=1)
    class_names = 0,1

    # Compute confusion matrix
    cm = compute_confusion_matrix(true_labels=labels, predicted_labels=predicted_labels, 
                                labels=class_names)
    log_confusion_matrix(cm, class_names)
    return compute_metrics_train_model(
                                        predicted_scores=predicted_scores,
                                        predicted_labels=predicted_labels,
                                        labels=labels
                                        )

############MODEL
# set hyperparameters
run = wandb.init()
try:
    sweep_model_name = wandb.config._name_or_path
    learning_rate = wandb.config.learning_rate
    num_train_epochs = wandb.config.num_train_epochs
    weight_decay = wandb.config.weight_decay
    seed = wandb.config.seed # run 20 times with different, random seeds (via wandb, then automatically 20 runs)

except:
    sweep_model_name = 'distilbert-base-uncased'
    learning_rate = 1e-5 # changed bc accuracy was much better
    num_train_epochs = 3
    weight_decay = 0.005 # against overfitting
    seed = 42

model = AutoModelForSequenceClassification.from_pretrained(sweep_model_name)
tokenizer = AutoTokenizer.from_pretrained(sweep_model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding='max_length',max_length=256)


############PREPROCESSING
def encode(ds):
    return tokenizer(ds["text"],  truncation=True, padding="max_length")
ds = ds.map(encode, batched=True)
#print(ds[0])
ds_dict = ds.train_test_split(test_size=0.2,
                                    seed=seed,
                                    shuffle=True)  # seed is set here!

training_args = TrainingArguments(
    output_dir="model_mixed",
    learning_rate=learning_rate,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    evaluation_strategy = "epoch", #IntervalStrategy.STEPS, # changed from "epoch" for early stopping
    #eval_steps = 1406, #bc. batch(16)*gpu(2)*1406 = 45000 samples per epoch (complete dataset)
    metric_for_best_model = 'f1_macro',
    save_strategy="epoch",
    save_total_limit = 10,
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_dict["train"],
    eval_dataset=ds_dict["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=3)], #probably not needed with 3 epochs
)

trainer.train()
trainer.evaluate()
now = datetime.now()
timestamp = now.strftime("%m_%d_%H_%M")
# save model in /data/verena/models/ -> in the end just keep the best.
trainer.save_model(f'/data/verena/models/model_mixed_{timestamp}')

