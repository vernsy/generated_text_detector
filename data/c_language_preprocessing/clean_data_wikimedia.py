import re
import os
from bs4 import BeautifulSoup
import json
from typing import Union

# this is needed in order to find module setup
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from setup import config

PROJECT_PATH = config.PROJECT_PATH
## input
INPUT_FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIMEDIA = config.FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIMEDIA
INPUT_OS_PATH_HUMAN_WRITTEN_SCRAPED = PROJECT_PATH+config.OS_PATH_HUMAN_WRITTEN_SCRAPED
## output
OUTPUT_FILENAME_HUMAN_WRITTEN_CLEANED_WIKIMEDIA = config.FILENAME_HUMAN_WRITTEN_CLEANED_WIKIMEDIA
OUTPUT_OS_PATH_HUMAN_WRITTEN_CLEANED = PROJECT_PATH+config.OS_PATH_HUMAN_WRITTEN_CLEANED


def is_list_not_text(text: str) -> bool:
    text_len = len(text)
    if text.count('*')/text_len > 0.01:
        return True # if each 100th char is a '*' then it is probably a list of writings or poems
    return False

def is_code_not_text(text: str) -> bool:
    indicator = 0

    if 'style' in text:
        indicator += 1
    
    
    return False

def clean_text(text: str) -> Union[str,bool]:

    # Remove html-tags by using beautiful soup parser
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)

     # Remove newline characters
    text = text.replace('\n', ' ')
    
    ###################
    # Remove all kind of Links [[...]]
    allowed_characters = r'a-zA-Z0-9!.,"`\'\’#\(\)\{\}\-/\$öäüßÜÖÄ \n\[\]'  # mind the empty space in the end of the collection!

    ## Remove [[...]] whatever tag that is
    pattern = rf'\[\[([{allowed_characters}]+)\]\]'
    text = re.sub(pattern, lambda match: replace_link_by_linkname_function(match,1), text)

    ## Remove links [[...:...|...]]
    pattern = rf'\[\[([{allowed_characters}]+):([{allowed_characters}]+)\|([{allowed_characters}]+)\]\]'
    text = re.sub(pattern, lambda match: replace_link_by_linkname_function(match,3), text)

    ## Remove links [[...:...:...]]
    pattern = rf'\[\[([{allowed_characters}]+):([{allowed_characters}]+):([{allowed_characters}]+)\]\]'
    text = re.sub(pattern, lambda match: replace_link_by_linkname_function(match,3), text)
   
    ## Remove links [[...:...:...|...]]
    pattern = rf'\[\[([{allowed_characters}]+):([{allowed_characters}]+):([{allowed_characters}]+)\|([{allowed_characters}]+)\]\]'
    text = re.sub(pattern, lambda match: replace_link_by_linkname_function(match,4), text)

    ## Remove links [[...:...]] without alternative name
    pattern = rf'\[\[([{allowed_characters}]+):([{allowed_characters}]+)\]\]'
    text = re.sub(pattern, lambda match: replace_link_by_linkname_function(match,2), text)

    ## Remove links [[...|...]] 
    pattern = rf'\[\[([{allowed_characters}]+)\|([{allowed_characters}]+)\]\]'
    text = re.sub(pattern, lambda match: replace_link_by_linkname_function(match,2), text)

    ## Remove the rest, where patterns hadn't been detected
    pattern = r'\[\s*\[\s*.*?\s*\]\s*\]'
    text = re.sub(pattern, '', text)

    ###################
    # Remove curly braces {{...}}''ALTERNATIVE NAME'' by TITEL
    pattern_sequence_to_replace = r'\{\{(.*?)\}\}\\n\'\'(.*?)\'\'\,'
    text = re.sub(pattern_sequence_to_replace, lambda match: replace_metadata_by_article_title(match), text)

    # Remove curly braces {{...}}'''ALTERNATIVE NAME''' by TITEL
    pattern_sequence_to_replace = r'\{\{(.*?)\}\}\\n\'\'\'(.*?)\'\'\''
    text = re.sub(pattern_sequence_to_replace, lambda match: replace_metadata_by_article_title(match), text)

    # Remove {{..|..}},{{...}}
    pattern_sequence_to_replace = r'\{\s*\{\s*.*?\s*\}\s*\}'
    text = re.sub(pattern_sequence_to_replace, '', text)

    ###################
    # Remove  '','''  
    pattern = r'\'\'\''
    text = re.sub(pattern, '', text)

    pattern = r'\'\''
    text = re.sub(pattern, '', text)

    # Remove ==...==
    pattern_sequence_to_replace = r'\=\=.*?\=\='
    text = re.sub(pattern_sequence_to_replace, '', text)

    # remove ----
    text = re.sub(r'--+', '', text)

    ###################
    # replace variable $... by #NAME
    pattern = r'["„“]*\$\d+\S*'
    text = re.sub(pattern, '#NAME', text)

    # Remove {| ... |}
    pattern_sequence_to_replace = r'\{\|\s*.*?\s*\|\}'
    text = re.sub(pattern_sequence_to_replace, '', text)
    # Remove | ... |}} # some leftover product from crazy link formats
    pattern_sequence_to_replace = r'\|\s*.*?\s*\}\}'
    text = re.sub(pattern_sequence_to_replace, '', text)

    ###################
    if text: # check if there is any text left afterall

        text = text.strip()
            
        if len(text) < 150: # check if text is longer than 1000 characters (below it is most probably metadata)
            return False
        if is_list_not_text(text):
            return False

    return text

def replace_link_by_linkname_function(match: str, group_index: int) -> str:
    # Return the last part of the match
    return match.group(group_index)

def replace_metadata_by_article_title(match: str) -> str:
    content_curly_braces = match.group(1)
    pattern_to_extract_TITEL = r'TITEL=(.*?)\n*\|'# re.compile(,re.DOTALL)
    
    title_match = re.sub(
        pattern_to_extract_TITEL, 
        lambda match: replace_link_by_linkname_function(match,1), 
        content_curly_braces
        )
   
    if title_match:
        return title_match +','
    return '#NAME'

def clean_timestamp(timestamp_str: str) -> str:
    return timestamp_str[:10] # YYYY-MM-DD

def main():
    input_file_path = os.path.join(INPUT_OS_PATH_HUMAN_WRITTEN_SCRAPED, 
                                   INPUT_FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIMEDIA
                                )
    output_file_path = os.path.join(OUTPUT_OS_PATH_HUMAN_WRITTEN_CLEANED,
                                    OUTPUT_FILENAME_HUMAN_WRITTEN_CLEANED_WIKIMEDIA
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
                timestamp_string = clean_timestamp(timestamp_string)
                text = obj.get("text", None)

                text_or_false = clean_text(text)
                if not text_or_false: # false means it contains list, too short etc
                    continue # do not write it down

                text = text_or_false # because it cannot be false at this point

                line_to_write = {"timestamp": timestamp_string,"text": text}
                output_file.write(json.dumps(line_to_write, ensure_ascii=False) + '\n')
                

    print("Successfully cleaned data from wikimedia.")

if __name__ == "__main__":
    main()