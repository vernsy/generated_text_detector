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
from sklearn.calibration import calibration_curve

# this is needed in order to find module setup
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from setup import config

PROJECT_PATH = config.PROJECT_PATH

OUTPUT_OS_PATH_LOGGING_PERFORMANCE_METRICS = PROJECT_PATH + config.OS_PATH_LOGGING_PERFORMANCE_METRICS
OUTPUT_LOGGING_FILE_PERFORMANCE_METRICS_BASELINE = config.OUTPUT_LOGGING_FILE_PERFORMANCE_METRICS_BASELINE

def log_metrics_to_commandline(
        model,val_loss, accuracy_value,precision_value_macro, recall_value_macro,
        f1_value_macro, f1_value_micro, mcc_value, conf_matrix_digits, class_report, #auroc_value, 
        ):

    print("OUTPUT FROM EVALUATION FUNCTION:")
    print("\nModel:",model)
    print("Validation Loss:",val_loss)
    print("Scikit-learn Accuracy:", accuracy_value)
    print("Scikit-learn Precision:", precision_value_macro)
    print("Scikit-learn Recall:", recall_value_macro)
    print("Scikit-learn F1 Score (Macro):", f1_value_macro)
    print("Scikit-learn F1 Score (Micro):", f1_value_micro)
    print("Scikit-learn Matthews Correlation Coefficient (MCC):", mcc_value)
    #print("Scikit-learn AUROC:", auroc_value)
    print("\nScikit-learn Confusion Matrix:\n", conf_matrix_digits)
    print("\nScikit-learn Classification Report:")
    print(class_report)

    return

def log_metrics_to_file(model, val_loss, accuracy_value, precision_value_macro, recall_value_macro,
                        f1_value_macro, f1_value_micro, mcc_value) -> None: #, auroc_value
    
    output_file_path = os.path.join(OUTPUT_OS_PATH_LOGGING_PERFORMANCE_METRICS,
                                    OUTPUT_LOGGING_FILE_PERFORMANCE_METRICS_BASELINE) 
    
    header = "Model\tLearningrate\tAccuracy\tPrecision\tRecall\tF1 Score Macro\tF1 Score Micro\tMCC\n" #\tAUROC

    # Check if the file already exists
    file_exists = os.path.exists(output_file_path)
    with open(output_file_path, 'a') as f_out:

        if not file_exists:
            f_out.write(header)
        

        line = f"{model}\t{val_loss:.5f}\t{accuracy_value:.4f}\t \
        {precision_value_macro:.4f}\t{recall_value_macro:.4f}\t{f1_value_macro:.4f}\t \
        {f1_value_micro:.4f}\t{mcc_value:.4f}\n" #\t{auroc_value:.4f}
        
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

def plot_calibration(true_labels, predicted_scores, ece_value, model_name):

    predicted_scores = predicted_scores[:,1]
    # keep indices when sorting predictions, in order to sort labels accordingly
    predicted_scores_sorted_indices = np.argsort(predicted_scores)
    true_labels_sorted = true_labels[predicted_scores_sorted_indices]
    predicted_scores_sorted = predicted_scores[predicted_scores_sorted_indices]

    bins = np.arange(0,1.1,0.1)
    counts, _ = np.histogram(predicted_scores_sorted, bins=bins)
    conf_values_per_bin = np.zeros(10)
    acc_values_per_bin = np.zeros(10)

    j = 0
    for i in range(len(counts)):
        preds_bin = predicted_scores_sorted[j : j+counts[i]]
        labels_bin = true_labels_sorted[j : j+counts[i]]
        conf_values_per_bin[i] = np.mean(preds_bin)
        acc_values_per_bin[i] = np.sum(labels_bin) / counts[i]
        j = j+counts[i]  # update j to the end of the current interval
        # print(f"################ bin {i}")
        # print("preds_per_bin", preds_bin)
        # print("labels_per_bin", labels_bin)
        # print("len_predicted_scores_complete: ", len(predicted_scores_sorted))
        # print("len bin and counts at index i", len(preds_bin),len(labels_bin), counts[i])
        # print(f"interval thresholds are {j} to {j+counts[i]}")
        
           
    print(f"Gap-size acc-conf for model {model_name}:",np.round(np.abs(acc_values_per_bin -conf_values_per_bin),2))

    # counts will contain the count of elements in each bin
    print("Number of elements in bin",counts)

    # Create the bar chart
    plt.figure(figsize=(8, 6))
    bar_width = 0.07
    beautiful_violet = (138/255, 43/255, 226/255, 0.3)
    beautiful_spring_green = (0, 255/255, 127/255, 0.3)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal 
    plt.bar(bins[:-1] + bar_width/1.5, conf_values_per_bin, width=bar_width, label='Confidence', color=beautiful_violet,bottom=0)
    plt.bar(bins[:-1] + bar_width/2, acc_values_per_bin, width=bar_width, label='Accuracy',color=beautiful_spring_green,bottom=0)

    # Add labels and title
    plt.xlabel('Intervals')
    plt.ylabel('Confidence and Accuracy for the positive Class (1 = machine generated)')
    plt.title(f'Calibration of the Model Trained on {model_name} Dataset')
    plt.xticks(bins[:-1]+0.1)

    # Add legend
    plt.legend(fontsize='large')
    beautiful_violet = (138/255, 43/255, 226/255,1)
    plt.text(0.65, 0.1, f'ECE: {ece_value:.4f}', fontsize=20, color=beautiful_violet)
    for i, (interval, value1, value2) in enumerate(zip(bins, conf_values_per_bin, acc_values_per_bin)):
        plt.text(interval + bar_width/2, max(value1, value2) + 0.02, f'{counts[i]}', ha='center')

    # Show the plot
    plt.savefig(f'calibration_bar_{model_name}.png')


def test_model_performance( model: str, 
                            y_true_np: np.ndarray, 
                            y_pred_np: np.ndarray, 
                            y_scores_np: np.ndarray,
                            val_loss: float,
                            ece_value: float,
                            ) -> Dict:


        accuracy_value = accuracy_score(y_true_np, y_pred_np)
        precision_value_macro = precision_score(y_true_np, y_pred_np, average='macro',zero_division=1)
        recall_value_macro = recall_score(y_true_np, y_pred_np, average='macro',zero_division=1)
        f1_value_macro = f1_score(y_true_np, y_pred_np, average='macro',zero_division=1)
        f1_value_micro = f1_score(y_true_np, y_pred_np, average='micro',zero_division=1)
        mcc_value = matthews_corrcoef(y_true_np, y_pred_np)

        ########This is for the AUROC value
        # shape was (2000,2)
        # probabilities were summing up to 1, so (1-p),p
        # the positive class is in the second column
        y_scores_class_1 = y_scores_np[:, 1]
        y_scores_class_0 = y_scores_np[:, 0]

        #auroc_value = roc_auc_score(y_true_np, y_scores_positive_class)

        ##confusion matrix
        conf_matrix_digits = confusion_matrix(y_true_np, y_pred_np)
        
        ##classification report:
        class_report = classification_report(y_true_np, y_pred_np)

        # Logging:
        log_metrics_to_file(model, val_loss, accuracy_value, precision_value_macro, recall_value_macro,
                        f1_value_macro, f1_value_micro, mcc_value) #auroc_value,
        log_metrics_to_commandline(model,val_loss, accuracy_value,precision_value_macro, recall_value_macro,
                        f1_value_macro, f1_value_micro, mcc_value,  conf_matrix_digits, class_report) #auroc_value,
        
        plot_calibration(y_true_np, y_scores_np, ece_value=ece_value, model_name=model)

        metrics_dict = {
            'accuracy': accuracy_value,
            'precision': precision_value_macro,
            'recall': recall_value_macro,
            'f1_macro': f1_value_macro,
            'f1_micro': f1_value_micro,
            'mcc': mcc_value,
            #'auroc': auroc_value,
            'conf_matrix': conf_matrix_digits,
        }
        return metrics_dict