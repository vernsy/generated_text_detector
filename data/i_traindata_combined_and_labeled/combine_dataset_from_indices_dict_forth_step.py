import json
import os
from typing import List, Tuple
import re
import csv
import sys
# make it possible to find the data python module in parent folder
sys.path.insert(0, '../..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.i_traindata_combined_and_labeled.util_i import *

NUMBER_FAILURES=0
LABEL_HUMAN_WRITTEN=0
LABEL_MACHINE_GENERATED=1


def load_json(FILENAME):
    path_object = os.path.join(OS_PATH_STATISTICS_OUTPUT,FILENAME)
    with open(path_object, 'r') as file:
        indices_per_quartile_dict = json.load(file)
    return indices_per_quartile_dict


def extract_as_string(input_string: str)-> str:
    
    pattern = r'llama_text[\'"]: [\'"](.*?)[\'"], [\'"]gen_text_len[\'"]:'

    # Use re.search to find the first match
    match = re.search(pattern, input_string)

    # Check if a match is found
    if match:
        # Extract the content between 'llama_generated: ' and ', 'gen_text_len':'
        extracted_text = match.group(1)
        #print("Extracted text:", extracted_text)
        try:
            extracted_text = extracted_text.encode('latin-1').decode('utf-8')
        except:
            num_fail = os.environ.get('NUMBER_FAILURES','0')
            os.environ['NUMBER_FAILURES'] = str(int(num_fail)+1)
            return None
    else:
        # No match found. ERROR! LLaMA sometimes wrote down stuff a bit weird...
        num_fail = os.environ.get('NUMBER_FAILURES','0')
        os.environ['NUMBER_FAILURES'] = str(int(num_fail)+1)
        return None
    
    pattern2 = r'\{[\'"]prompt[\'"]: [\'"](.*?)[\'"], [\'"]llama_text[\'"]: [\'"]'
    extracted_text_no_prompt = re.sub(pattern2, '', extracted_text)
    #print("Extracted text:", extracted_text_no_prompt)
    return extracted_text_no_prompt

def extract_from_tsv(tsv_file_path: str) -> List[str]:
    first_column = []
    with open(tsv_file_path, 'r', newline='', encoding='utf-8') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            if row:  # Check if the row is not empty
                first_column.append(row[0])
                print(row[0][:19])
    return first_column

def get_indices_per_quartiles_as_integer_list(quartile_indices_dict):
    list_list = []
    for quartile ,values in quartile_indices_dict.items():
        int_values = [int(digit) for digit in values]
        list_list += [int_values]
    assert len(list_list)==4, "Apparently not exactly 4 quartiles!!!!"
    return list_list

def load_human_written_dataset(indices_dict) -> Tuple[List[str], List[bool]]:
    texts = []
    labels = []

    for key, quartile_indices_dict in indices_dict.items():
        [q0, q1, q2, q3] = get_indices_per_quartiles_as_integer_list(quartile_indices_dict)
        filename = key[len("token_lens_"):]

        human_written_input = os.path.join(
            OS_PATH_HUMAN_WRITTEN_CLEANED,
            filename
        )

        with open(human_written_input, 'r',encoding='utf-8') as human_written_input_file:

            for i, line in enumerate(human_written_input_file):
                if not any(i in q for q in [q0,q1,q2,q3]):
                    continue

                try:
                    obj = json.loads(line[:-2]) # do not load the comma +\n that scrapy adds by default
                except:
                    try:
                        obj = json.loads(line[:-1]) # probably last line
                    except: continue

                text = obj.get("text", None) # all files have "text" key
                label = LABEL_HUMAN_WRITTEN           
                texts.append(text)
                labels.append(label)

    return texts, labels

def load_machine_generated_dataset_llama(indices_dict,
                                         PATH=OS_PATH_MACHINE_GENERATED_TRAININGDATA_LLAMA,
                                         encoding='latin-1',
                                         is_llama= bool) -> Tuple[List[str], List[bool]]:
    texts = []
    labels = []

    for key, quartile_indices_dict in indices_dict.items():
        [q0, q1, q2, q3] = get_indices_per_quartiles_as_integer_list(quartile_indices_dict)

        filename = key[len("token_lens_"):]

        machine_generated_input_llama = os.path.join(
            PATH, 
            filename
            )

        with open(machine_generated_input_llama, 'r', encoding=encoding) as machine_generated_input_file:
      
            for i, line in enumerate(machine_generated_input_file.readlines()):

                if not any(i in q for q in [q0,q1,q2,q3]):
                    continue
                if not is_llama:
                    columns = line.strip().split('\t')
                    text = columns[0]
                elif is_llama:
                    text = extract_as_string(line)
                label = LABEL_MACHINE_GENERATED
                texts.append(text)
                labels.append(label)
        
    return texts, labels

def main(mixed: bool) -> Tuple[List[str], List[bool]]:
    # load jsons
    # train_eval_indices_per_quartile_dict_hw = load_json(FILENAME_JSON_INDICES_DICT_HW_TRAIN_EVAL)
    # train_eval_indices_per_quartile_dict_mg_llama = load_json(FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_LLAMA)
    train_eval_indices_per_quartile_dict_mg_mixed = load_json(FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_MIXED)

    # question: do the entries have to be mixed by label? -> No, it is done in the DataLoader
    texts=[]
    labels=[]

    # hw_texts, hw_labels = load_human_written_dataset(train_eval_indices_per_quartile_dict_hw)
    # mg_texts, mg_labels = load_machine_generated_dataset_llama(train_eval_indices_per_quartile_dict_mg_llama)
    mixed_texts, mixed_labels = load_machine_generated_dataset_llama(train_eval_indices_per_quartile_dict_mg_mixed, 
                                                                     PATH=OS_PATH_MG_DATA_WIZARD_SNOOZY_MISTRAL_GPT,
                                                                     encoding='utf-8',
                                                                     is_llama=False)

    # texts = hw_texts + mg_texts + mixed_texts
    # labels = hw_labels + mg_labels + mixed_labels

    # hw_path_object = os.path.join(OS_PATH_DATASET_TEXTS_ONLY,FILENAME_HW_DATASET_TEXTS_TRAIN_EVAL)
    # with open(hw_path_object, 'w', encoding='utf-8') as out_hw:
    #     for entry in hw_texts:
    #         json.dump(entry, out_hw,ensure_ascii=False)  # Write the JSON object
    #         out_hw.write('\n')        # Add a newline character

    # mg_path_object = os.path.join(OS_PATH_DATASET_TEXTS_ONLY,FILENAME_MG_DATASET_TEXTS_TRAIN_EVAL_LLAMA)
    # with open(mg_path_object, 'w', encoding='utf-8') as out_mg:
    #     for entry in mg_texts:
    #         json.dump(entry, out_mg,ensure_ascii=False)  # Write the JSON object
    #         out_mg.write('\n')        # Add a newline character

    mixed_path_object = os.path.join(OS_PATH_DATASET_TEXTS_ONLY,FILENAME_MG_DATASET_TEXTS_TRAIN_EVAL_MIXED)
    with open(mixed_path_object, 'w', encoding='utf-8') as out_mixed:
        for entry in mixed_texts:
            json.dump(entry, out_mixed,ensure_ascii=False)  # Write the JSON object
            out_mixed.write('\n')        # Add a newline character

    print("Length of mixed texts:",len(mixed_texts)," Length of labels:",len(mixed_labels))
    print("Length of texts:",len(texts)," Length of labels:",len(labels))
    print("Labels are equally used: \nOverall Sum = 0.5*len(labels):",sum(labels))
    print("Number of failures due to format mismatches in LLaMA output:", os.environ.get('NUMBER_FAILURES'))


    print("\nSuccessfully combined the dataset.")
    return texts,labels
    #return

if __name__ == "__main__":
    main(mixed=True)

