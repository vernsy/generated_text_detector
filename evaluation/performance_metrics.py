from datetime import datetime
import os
from typing import Dict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
                            f1_score, confusion_matrix, classification_report,\
                            roc_auc_score, matthews_corrcoef

# this is needed in order to find module setup
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from setup import config

PROJECT_PATH = config.PROJECT_PATH

OUTPUT_OS_PATH_LOGGING_PERFORMANCE_METRICS = PROJECT_PATH + config.OS_PATH_LOGGING_PERFORMANCE_METRICS
OUTPUT_LOGGING_FILE_PERFORMANCE_METRICS_BASELINE = config.OUTPUT_LOGGING_FILE_PERFORMANCE_METRICS_BASELINE

def log_metrics_to_commandline(
        timestamp,model,epoch_score,learning_rate, train_loss, val_loss, accuracy_value,precision_value_macro, recall_value_macro,
        f1_value_macro, f1_value_micro, mcc_value, auroc_value, conf_matrix_digits, class_report,
        ):

    print("OUTPUT FROM EVALUATION FUNCTION:")
    print("Timestamp:", timestamp)
    print("\nModel:",model)
    print("Epoch:",epoch_score)
    print("Learning-rate:", learning_rate)
    print("Training Loss:",train_loss)
    print("Validation Loss:",val_loss)
    print("Scikit-learn Accuracy:", accuracy_value)
    print("Scikit-learn Precision:", precision_value_macro)
    print("Scikit-learn Recall:", recall_value_macro)
    print("Scikit-learn F1 Score (Macro):", f1_value_macro)
    print("Scikit-learn F1 Score (Micro):", f1_value_micro)
    print("Scikit-learn Matthews Correlation Coefficient (MCC):", mcc_value)
    print("Scikit-learn AUROC:", auroc_value)
    print("\nScikit-learn Confusion Matrix:\n", conf_matrix_digits)
    print("\nScikit-learn Classification Report:")
    print(class_report)

    return

def log_metrics_to_file(model, epoch_score, timestamp, learning_rate, train_loss, val_loss, accuracy_value, precision_value_macro, recall_value_macro,
                        f1_value_macro, f1_value_micro, mcc_value, auroc_value) -> None:
    
    output_file_path = os.path.join(OUTPUT_OS_PATH_LOGGING_PERFORMANCE_METRICS,
                                    OUTPUT_LOGGING_FILE_PERFORMANCE_METRICS_BASELINE) 
    
    header = "Model\tEpoch\tTimestamp\tLearningrate\tTrain Loss\tValidation Loss\tAccuracy\tPrecision\tRecall\tF1 Score Macro\tF1 Score Micro\tMCC\tAUROC\n"

    # Check if the file already exists
    file_exists = os.path.exists(output_file_path)
    with open(output_file_path, 'a') as f_out:

        if not file_exists:
            f_out.write(header)
        

        line = f"{model}\t{epoch_score}\t{timestamp}\t{learning_rate:.5f}\t{train_loss:.5f}\t{val_loss:.5f}\t{accuracy_value:.4f}\t \
        {precision_value_macro:.4f}\t{recall_value_macro:.4f}\t{f1_value_macro:.4f}\t \
        {f1_value_micro:.4f}\t{mcc_value:.4f}\t{auroc_value:.4f}\n"
        
        f_out.write(line)

    return

def plot_confusion_matrix(confusion_matrix_digits):
     # Plot confusion matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_digits, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    #plt.show()

    return plt

def evaluate_baseline_model_performance( model: str, 
                                        epoch: int,
                                        num_epochs: int, 
                                        y_true_np: np.ndarray, 
                                        y_pred_np: np.ndarray, 
                                        y_scores_np: np.ndarray,
                                        train_loss: float,
                                        val_loss: float,
                                        learning_rate: float,
                                        ) -> Dict:


        # get current timestamp and metric values
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        epoch_score = f'{epoch + 1}/{num_epochs}'
        accuracy_value = accuracy_score(y_true_np, y_pred_np)
        precision_value_macro = precision_score(y_true_np, y_pred_np, average='macro')
        recall_value_macro = recall_score(y_true_np, y_pred_np, average='macro')
        f1_value_macro = f1_score(y_true_np, y_pred_np, average='macro')
        f1_value_micro = f1_score(y_true_np, y_pred_np, average='micro')
        mcc_value = matthews_corrcoef(y_true_np, y_pred_np)

        ########This is for the AUROC value
        # shape was (2000,2)
        # probabilities were summing up to 1, so (1-p),p
        # the positive class is in the second column
        y_scores_positive_class = y_scores_np[:, 1]
        auroc_value = roc_auc_score(y_true_np, y_scores_positive_class)

        ##confusion matrix
        conf_matrix_digits = confusion_matrix(y_true_np, y_pred_np)
        
        ##classification report:
        class_report = classification_report(y_true_np, y_pred_np)

        # Logging:
        log_metrics_to_file(model, epoch_score, timestamp_str, learning_rate, train_loss, val_loss, accuracy_value, precision_value_macro, recall_value_macro,
                        f1_value_macro, f1_value_micro, mcc_value, auroc_value)
        log_metrics_to_commandline(timestamp_str,model,epoch_score, learning_rate, train_loss, val_loss, accuracy_value,precision_value_macro, recall_value_macro,
                        f1_value_macro, f1_value_micro, mcc_value, auroc_value, conf_matrix_digits, class_report)

        metrics_dict = {
            'timestamp': timestamp,
            'epoch': epoch,
            'accuracy': accuracy_value,
            'precision': precision_value_macro,
            'recall': recall_value_macro,
            'f1_macro': f1_value_macro,
            'f1_micro': f1_value_micro,
            'mcc': mcc_value,
            'auroc': auroc_value,
            'conf_matrix': conf_matrix_digits,
            'class_report': class_report
        }
        return metrics_dict


def compute_metrics_train_model(predicted_scores,
                                predicted_labels,
                                labels
                                ) -> Dict:


        # get metric values
        accuracy_value = accuracy_score(labels, predicted_labels)
        #accuracy_value = accuracy_score(y_true_np, y_pred_np)
        precision_value_macro = precision_score(labels, predicted_labels, average='macro')
        recall_value_macro = recall_score(labels, predicted_labels, average='macro')
        f1_value_macro = f1_score(labels, predicted_labels, average='macro')
        f1_value_micro = f1_score(labels, predicted_labels, average='micro')
        mcc_value = matthews_corrcoef(y_true=labels, y_pred=predicted_labels)

        ########This is for the AUROC value
        # shape was (2000,2)
        # probabilities were summing up to 1, so (1-p),p
        # the positive class is in the second column
        y_scores_positive_class = predicted_scores[:, 1]
        auroc_value = roc_auc_score(y_true=labels,y_score=y_scores_positive_class)

        ##confusion matrix
        conf_matrix_digits = confusion_matrix(y_true=labels, y_pred=predicted_labels)
        
        ##classification report:
        class_report = classification_report(y_true=labels, y_pred=predicted_labels)

        
        metrics_dict = {
            'accuracy': accuracy_value,
            'precision': precision_value_macro,
            'recall': recall_value_macro,
            'f1_macro': f1_value_macro,
            'f1_micro': f1_value_micro,
            'mcc': mcc_value,
            'auroc': auroc_value,
            'conf_matrix': conf_matrix_digits.tolist(),
        }
        return metrics_dict