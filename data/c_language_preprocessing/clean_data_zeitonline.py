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
INPUT_FILENAME_HUMAN_WRITTEN_RAW_ZEITONLINE = config.FILENAME_HUMAN_WRITTEN_RAW_ZEITONLINE
INPUT_OS_PATH_HUMAN_WRITTEN_RAW = PROJECT_PATH + config.OS_PATH_HUMAN_WRITTEN_RAW
## output
OUTPUT_FILENAME_HUMAN_WRITTEN_CLEANED_ZEITONLINE = config.FILENAME_HUMAN_WRITTEN_CLEANED_ZEITONLINE
OUTPUT_OS_PATH_HUMAN_WRITTEN_CLEANED = PROJECT_PATH + config.OS_PATH_HUMAN_WRITTEN_CLEANED

def check_for_creation_date(timestamp_string: str) -> Union[bool, str]:
    locale.setlocale(locale.LC_TIME, 'de_DE.utf-8')
    date_format = "%d. %B %Y"
    try: # sometimes the time is 'vor 17 Stunden' which will throw an error, hence the try
        timestamp = datetime.strptime(timestamp_string, date_format)
    except:
        return False

    # texts should be older than 2016
    threshold_timestamp = datetime(2016, 1, 1)
    if timestamp > threshold_timestamp:
        return False
    return timestamp.strftime("%Y-%m-%d")

def main():
    input_file_path = os.path.join(INPUT_OS_PATH_HUMAN_WRITTEN_RAW, 
                                   INPUT_FILENAME_HUMAN_WRITTEN_RAW_ZEITONLINE
                                )
    output_file_path = os.path.join(OUTPUT_OS_PATH_HUMAN_WRITTEN_CLEANED,
                                    OUTPUT_FILENAME_HUMAN_WRITTEN_CLEANED_ZEITONLINE
                                    )
        
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                obj = json.loads(line)
                ## keys: 0=author, 1=time, 2=topic, 3=comment
                timestamp_string = obj.get("time", None)
                timestamp_string_or_false = check_for_creation_date(timestamp_string)
                if not timestamp_string_or_false: # text too young
                    continue
                text = obj.get("comment", None)
                # Remove newline characters
                text = text.replace('\n', ' ')

                line_to_write = {"timestamp": timestamp_string_or_false,"text": text}
                output_file.write(json.dumps(line_to_write, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()