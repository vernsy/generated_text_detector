import os
import json
from datetime import datetime
import locale
from typing import Union
import re

# this is needed in order to find module setup
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from setup import config

PROJECT_PATH = config.PROJECT_PATH
## input
INPUT_FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIPEDIA = config.FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIPEDIA
INPUT_OS_PATH_HUMAN_WRITTEN_SCRAPED = PROJECT_PATH + config.OS_PATH_HUMAN_WRITTEN_SCRAPED
## output
OUTPUT_FILENAME_HUMAN_WRITTEN_CLEANED_WIKIPEDIA = config.FILENAME_HUMAN_WRITTEN_CLEANED_WIKIPEDIA
OUTPUT_OS_PATH_HUMAN_WRITTEN_CLEANED = PROJECT_PATH + config.OS_PATH_HUMAN_WRITTEN_CLEANED

# creation date does not need to be checked, because dump is from 2008
def convert_all_dates(match) -> str:
    locale.setlocale(locale.LC_TIME, 'de_DE.utf-8')
    date_format = "%d. %B %Y"
    timestamp_string = match.group(0)
    try: # sometimes the spaces differ, or typos
        timestamp = datetime.strptime(timestamp_string, date_format)
    except:
        return timestamp_string

    return timestamp.strftime("%Y-%m-%d")

def clean_text(text: str) -> Union[str,bool]:
    # ignore code
    if contains_code(text):
        return False
    
    # clean text when not code
    text = text.replace('\n', ' ')
    text = text.replace('[ Bearbeiten ]','')
    text = re.sub(r'<!.*?>','', text)
    # two or more empty spaces to one
    text = re.sub(r'\s+', ' ', text)
    # delete too many spaces before special characters
    text = re.sub(r'\s*([.,;:)!])', r'\1', text)
    text = re.sub(r'([(])\s*', r'\1', text)
    text = text.replace('↑','')
    text = text.replace('//','')
    text = text.replace('~~','')
    text = text.replace('--','')
    # convert dates bc then they won't be cut by sentence tokenizer
    date_pattern = r'\b\d{1,2}\. [A-Za-zäöüßÄÖÜéè]+ \d{4}\b'
    text = re.sub(date_pattern, convert_all_dates, text)

    return text

def contains_code(text: str) -> bool:
    indicator = 0
    # Check for code indicators
    if ' if ' in text:
        indicator += 1
    if ' var ' in text:
        indicator += 1
    if ' // ' in text:
        indicator += 1
    if '@param' in text:
        indicator +=1
    
    if indicator >= 4:
        return True
    
    return False

def main():
    input_file_path = os.path.join(INPUT_OS_PATH_HUMAN_WRITTEN_SCRAPED, 
                                   INPUT_FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIPEDIA
                                )
    output_file_path = os.path.join(OUTPUT_OS_PATH_HUMAN_WRITTEN_CLEANED,
                                    OUTPUT_FILENAME_HUMAN_WRITTEN_CLEANED_WIKIPEDIA
                                    )
        
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                
                try:
                    obj = json.loads(line[:-2]) # do not load the comma +\n that scrapy adds by default
                except:
                    try:
                        obj = json.loads(line[:-1]) # probably last line
                    except: continue
                ## keys: 0=timestamp, 1=text
                timestamp_string = obj.get("timestamp", None)
                text = obj.get("text", None)

                text_or_false = clean_text(text)
                if not text_or_false: # false means it contains too much code
                    continue # do not write it down

                text = text_or_false # because it cannot be false at this point

                line_to_write = {"timestamp": timestamp_string,"text": text}
                output_file.write(json.dumps(line_to_write, ensure_ascii=False) + '\n')

    print("Successfully cleaned data from wikipedia.")

if __name__ == "__main__":
    main()