import os
import json
from datetime import datetime
import locale
from typing import Union
import re

## for language detection, kick out english text:
from langdetect import detect
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

from util import INPUT_OS_PATH_HUMAN_WRITTEN_SCRAPED, INPUT_FILENAME_HUMAN_WRITTEN_SCRAPED_SPRINGERNATURE,\
                OUTPUT_OS_PATH_HUMAN_WRITTEN_CLEANED, OUTPUT_FILENAME_HUMAN_WRITTEN_CLEANED_SPRINGERNATURE


def remove_english_sentences(text: str) -> str:
    sentences = sent_tokenize(text)

    non_english_sentences = []
    for sentence in sentences:
        # Check if the sentence is long enough for language detection
        if len(sentence) > 20:
            try:
                detected_language = detect(sentence)
                if detected_language != 'en':
                    non_english_sentences.append(sentence)
            except:
                # this applies mostly for URLs, so no problem
                print("Deleted, because could not detect language of:", sentence)
        else:
            non_english_sentences.append(sentence.strip())
    result_text = ' '.join(non_english_sentences)
    return result_text

def check_for_creation_date(timestamp_string: str) -> Union[bool, str]:
    date_format = "%Y-%m-%d"
    try:
        timestamp = datetime.strptime(timestamp_string, date_format)
    except:
        print("Timestamp not detected")
        return False

    # texts should be older than 2016
    threshold_timestamp = datetime(2016, 1, 1)
    if timestamp > threshold_timestamp:
        return False
    return timestamp.strftime("%Y-%m-%d")

def main():
    input_file_path = os.path.join(INPUT_OS_PATH_HUMAN_WRITTEN_SCRAPED, 
                                   INPUT_FILENAME_HUMAN_WRITTEN_SCRAPED_SPRINGERNATURE
                                   )
    output_file_path = os.path.join(OUTPUT_OS_PATH_HUMAN_WRITTEN_CLEANED,
                                    OUTPUT_FILENAME_HUMAN_WRITTEN_CLEANED_SPRINGERNATURE
                                    )
        
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            data = input_file.read()           
            json_objects = data.split('},\n')

            for line in json_objects:
                line = line + '}'
                line = line.replace('\xa0', ' ') # springer nature uses \xa0 to make symbols stick together
                if len(line) < 20:
                    continue
           
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(e)
                    continue
                
                ## keys: 0=publicationDate, 1=abstract
                timestamp_string = obj.get("publicationDate", None)
                timestamp_string_or_false = check_for_creation_date(timestamp_string)

                if not timestamp_string_or_false: # text too young
                    continue
                text_maybe_with_english = obj.get("abstract", None)
                text = remove_english_sentences(text_maybe_with_english)

                if len(text) > 20:
                    line_to_write = {"timestamp": timestamp_string_or_false,"text": text}
                    output_file.write(json.dumps(line_to_write, ensure_ascii=False) + '\n')

    print("Cleaned springernature data succesfully.")

if __name__ == "__main__":
    main()