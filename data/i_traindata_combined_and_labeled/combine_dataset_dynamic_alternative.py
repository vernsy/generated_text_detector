import json
import os
from typing import List, Tuple
import re
import csv
import random
import sys
# make it possible to find the data python module in parent folder
sys.path.insert(0, '..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .util_i import human_written_files, machine_generated_files, \
                    OS_PATH_MACHINE_GENERATED_TRAININGDATA_LLAMA, \
                    OS_PATH_HUMAN_WRITTEN_CLEANED

# All numbers correspond to entries, not byte sizes
NUMBER_ENTRIES_DATASET = 10000
NUMBER_ENTRIES_PER_CLASS = NUMBER_ENTRIES_DATASET/2
NUM_GENRES = 5
NUMBER_OF_ENTRIES_HW_PER_GENRE=NUMBER_ENTRIES_PER_CLASS/NUM_GENRES
NUMBER_OF_ENTRIES_MG_PER_GENRE=NUMBER_ENTRIES_PER_CLASS/NUM_GENRES

MODEL_FACTOR_LLAMA = 1
MODEL_FACTOR_MISTRAL = 0
MODEL_FACTOR_GPT = 0
MODEL_FACTOR_SNOOZY = 0
MODEL_FACTOR_WIZARD = 0

NUMBER_OF_ENTRIES_FROM_MODEL_LLAMA_PER_GENRE = MODEL_FACTOR_LLAMA*NUMBER_OF_ENTRIES_MG_PER_GENRE
NUMBER_OF_ENTRIES_FROM_MODEL_GPT_PER_GENRE = MODEL_FACTOR_GPT*NUMBER_OF_ENTRIES_MG_PER_GENRE
NUMBER_OF_ENTRIES_FROM_MODEL_WIZARD_PER_GENRE = MODEL_FACTOR_WIZARD*NUMBER_OF_ENTRIES_MG_PER_GENRE
NUMBER_OF_ENTRIES_FROM_MODEL_SNOOZY_PER_GENRE = MODEL_FACTOR_SNOOZY*NUMBER_OF_ENTRIES_MG_PER_GENRE
NUMBER_OF_ENTRIES_FROM_MODEL_MISTRAL_PER_GENRE = MODEL_FACTOR_MISTRAL*NUMBER_OF_ENTRIES_MG_PER_GENRE

PORTION_EVALUATION_DATASET = 0.2
PORTION_TEST_DATASET = 0.1 # split by indices dict!

NUMBER_FAILURES=0
LABEL_HUMAN_WRITTEN=0
LABEL_MACHINE_GENERATED=1


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

def main() -> Tuple[List[str], List[bool]]:
    # question: do the entries have to be mixed by label? -> No, it is done in the DataLoader
    texts=[]
    labels=[]

    for index_genre,HUMAN_WRITTEN_FILE in enumerate(human_written_files):
        human_written_input = os.path.join(
            OS_PATH_HUMAN_WRITTEN_CLEANED,
            HUMAN_WRITTEN_FILE
        )

        with open(human_written_input, 'r',encoding='utf-8') as human_written_input_file:
            #print(human_written_input_file)
            #human_written_input_json = json.load(human_written_input_file)
            counter = 0
            for line in human_written_input_file:
                if counter >= NUMBER_OF_ENTRIES_HW_PER_GENRE:
                    break

                try:
                    obj = json.loads(line[:-2]) # do not load the comma +\n that scrapy adds by default
                except:
                    try:
                        obj = json.loads(line[:-1]) # probably last line
                    except: continue

                text = obj.get("text", None) # all files have "text" key
                if not text:
                    continue
                label = LABEL_HUMAN_WRITTEN
                
                texts = texts + [text]
                labels.append(label)
                counter += 1

            print(len(texts),"=?=", (index_genre+1),"*",NUMBER_OF_ENTRIES_HW_PER_GENRE)
            #assert len(texts)==(index_genre+1)*NUMBER_OF_ENTRIES_PER_CLASS_OF_GENRE, "The limit did not work!"

    for MACHINE_GENERATED_FILE in machine_generated_files:

        machine_generated_input = os.path.join(
            OS_PATH_MACHINE_GENERATED_TRAININGDATA_LLAMA, 
            MACHINE_GENERATED_FILE
            )

        with open(machine_generated_input, 'r', encoding='latin-1') as machine_generated_input_file:
            counter=0            
            for line in machine_generated_input_file.readlines():
                text = extract_as_string(line)
                if text is None:
                    continue
                if counter >= NUMBER_OF_ENTRIES_FROM_MODEL_LLAMA_PER_GENRE:
                    break
                label = LABEL_MACHINE_GENERATED
                texts.append(text)
                labels.append(label)
                counter += 1

    #assert len(texts)==len(labels)==(index_genre+1)*NUMBER_OF_ENTRIES_PER_GENRE, "The numers of entries are not equal or not NUMBER_OF_ENTRIES_PER_GENRE long!"
    #assert len(texts)==len(labels), "The numers of entries are not equal"
    print("Length of texts:",len(texts)," Length of labels:",len(labels))
    print("Labels are equally used: 1. Overall Sum = 0.5*len(labels):",sum(labels),
          "check if 0:",sum(labels[:int(NUMBER_ENTRIES_PER_CLASS)]),
          "check if equals sum of total: ", sum(labels[int(NUMBER_ENTRIES_PER_CLASS):])
          )
    #print("Number of \"usable\" machine generated texts (originally):",len(machine_generated_texts))
    print("Number of failures due to format mismatches in LLaMA output:", os.environ.get('NUMBER_FAILURES'))
    # 315 format errors (no match) and on top 65 encoding errors -> it is OK, not too much lost
    print("Random controls: ")
    for i in range(0,5):
        idx=random.randint(0,2500-1)
        #print("text: ", texts[idx][:50],"...   Label:",labels[idx])
        #print("text: ", texts[idx+2500][:50],"...   Label:",labels[idx+2500])
    
    print("\nSuccessfully combined the dataset.")
    return texts,labels
    #return

if __name__ == "__main__":
    main()

