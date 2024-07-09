import re
import os

FILENAME = 'output_taz.json'

def preprocess_text(text):
    
    ## Remove links ]...[
    pattern =  '\]\s*\['
    text = re.sub(pattern,',', text)

    return text

def main():
    raw_text_file = os.path.join('01_data_unpreprocessed', FILENAME)
    with open(raw_text_file, 'r') as file:
        raw_text = file.read()

    preprocessed_text = preprocess_text(raw_text)

    output_text_file = os.path.join('02_data_cleaned', FILENAME)
    with open(output_text_file, 'w') as file:
        # Write content to the file
        file.write(preprocessed_text)

if __name__ == "__main__":
    main()