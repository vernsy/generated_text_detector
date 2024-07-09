from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
                        TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy
import torch
import evaluate
import numpy as np
import wandb
import sys
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


###########DATASET
# make it possible to find the data python module in parent folder
sys.path.insert(0, '..') 
# self written classes/modules
from data.i_traindata_combined_and_labeled import combine_dataset_dynamic_alternative
from evaluation.performance_metrics import compute_metrics_train_model
from datasets import Dataset
# Load dataset and split it into training and validation sets
texts, labels = combine_dataset_dynamic_alternative.main()
# load on GPU -> do not allow cpu, because when GPU is full it tries CPU and then it can't find all tensors
device = torch.device("cuda:0")#"cuda" if torch.cuda.is_available() else "cpu")
ds = Dataset.from_dict({"text": texts, "label": labels}).with_format("torch", device = device)
#print(ds[:3])

############MODEL
run = wandb.init()
try:
    sweep_model_name = wandb.config._name_or_path
    learning_rate = wandb.config.learning_rate
    num_train_epochs = wandb.config.num_train_epochs
    weight_decay = wandb.config.weight_decay
except:
    sweep_model_name = 'distilbert-base-uncased'
    learning_rate = 1e-5
    num_train_epochs = 10
    weight_decay = 0.005
model = AutoModelForSequenceClassification.from_pretrained(sweep_model_name)
tokenizer = AutoTokenizer.from_pretrained(sweep_model_name)


############PREPROCESSING
def encode(ds):
    return tokenizer(ds["text"],  truncation=True, padding="max_length")
ds = ds.map(encode, batched=True)
#print(ds[0])
ds_dict = ds.train_test_split(test_size=0.2,
                                    seed=42,
                                    shuffle=True)  # seed is set here!

###############EVALUATION
# Compute confusion matrix
def compute_confusion_matrix(true_labels, predicted_labels, labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    return cm

# Log confusion matrix to wandb
def log_confusion_matrix(confusion_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
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
    cm = compute_confusion_matrix(true_labels=labels, predicted_labels=predicted_labels, labels=class_names)
    log_confusion_matrix(cm, class_names)
    return compute_metrics_train_model(
                                        predicted_scores=predicted_scores,
                                        predicted_labels=predicted_labels,
                                        labels=labels
                                        )

training_args = TrainingArguments(
    output_dir="check_best_learningrate",
    learning_rate=learning_rate,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    evaluation_strategy = IntervalStrategy.STEPS, # changed from "epoch" for early stopping
    eval_steps = 250,
    metric_for_best_model = 'f1_macro',
    #save_strategy="epoch",
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
    #data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()