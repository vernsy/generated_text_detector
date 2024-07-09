import json
import os

# this is needed in order to find module setup
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from setup import config

PROJECT_PATH = config.PROJECT_PATH
## input
INPUT_FILENAME_HUMAN_WRITTEN_CLEANED = config.FILENAME_HUMAN_WRITTEN_CLEANED_WIKIMEDIA # replace genre here
INPUT_OS_PATH_HUMAN_WRITTEN_CLEANED = PROJECT_PATH + config.OS_PATH_HUMAN_WRITTEN_CLEANED
## output
OUTPUT_FILENAME_HUMAN_WRITTEN_PROMPTS = config.FILENAME_HUMAN_WRITTEN_PROMPTS_WIKIMEDIA # replace genre here
OUTPUT_OS_PATH_HUMAN_WRITTEN_PROMPTS = PROJECT_PATH + config.OS_PATH_HUMAN_WRITTEN_PROMPTS

FLAG_WIKIMEDIA = True

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

custom_abbreviations = [
    'z.B.', 'z. B.', 'd.h.', 's.u.', 'bzw.', 'u.a.', 'etc.', 'ca.', 'ugs.', 'i.d.R.', 'evtl.', 'u.v.m.', 'o.ä.',
    'z.T.', 's.o.', 'jmd.', 'etw.', 
    'Jan.', 'Feb.', 'Mrz.', 'Apr.', 'Jun.', 'Jul.', 'Aug.', 'Sept.', 'Okt.', 'Nov.', 'Dez.','Dec.',
    '0.', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.', '14.', '15.', '16.', 
    '17.', '18.', '19.', '20.', '21.', '22.', '23.', '24.', '25.', '26.', '27.', '28.', '29.', '30.', '31.', 
    'usw.', 'Rußl.', 'russ.', 'frz.', 'dt.', 'tschech.',
    'lat.','bayr.','ital.','pers.',
    'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.', 'J.', 'K.', 'L.', 'M.', 'N.', 'O.', 'P.', 'Q.', 'R.',
    'S.', 'T.', 'U.', 'V.', 'W.', 'X.', 'Y.', 'Z.',
    ' a.', ' b.', ' c.', ' d.', ' e.', ' f.', ' g.', ' h.', ' i.', ' j.', ' k.', ' l.', ' m.', ' n.', ' o.',
    ' p.', ' q.', ' r.', ' s.', ' t.', ' u.', ' v.', ' w.', ' x.', ' y.', ' z.',
    'spr.','Hg.','Hrg.', 'Verl.','Nr.',
    'v. Chr.', 'n. Chr.', 'jährl.', 'monatl.', 'tägl.', 'stündl.', 'ev.','geb.'
]

nltk.download('punkt')  # Download the punkt tokenizer models
punkt_params = PunktParameters()
punkt_params.abbrev_types = set(custom_abbreviations)
sentence_tokenizer = PunktSentenceTokenizer(punkt_params)

def cut_off_first_two_sentences(text: str) -> str:
    # Tokenize the text into sentences using the custom sentence tokenizer
    sentences = sentence_tokenizer.tokenize(text)

    if len(sentences) >= 2:
        # extra handle for wikimedia, as texts start at 
        # page breaks sometimes in the middle of a sentence
        if FLAG_WIKIMEDIA:
            first_sequence_until_dot = sentences[0].strip()
            if first_sequence_until_dot[0].islower():
                sentences = sentences[1:]
        # Cut off the first two sentences
        remaining_text = ' '.join(sentences[:2])
        return remaining_text.strip()
    else:
        return text.strip()

def main():
    # take untokenized input because less data to load
    input_file_path = os.path.join(INPUT_OS_PATH_HUMAN_WRITTEN_CLEANED, 
                                   INPUT_FILENAME_HUMAN_WRITTEN_CLEANED) 
    output_file_path = os.path.join(OUTPUT_OS_PATH_HUMAN_WRITTEN_PROMPTS,
                                   OUTPUT_FILENAME_HUMAN_WRITTEN_PROMPTS)

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in input_file:
                obj = json.loads(line)
                ## keys: 0=timestamp, 1=text
                text = obj.get("text", None)                                   
                text_length = len(text)
                two_sentences = cut_off_first_two_sentences(text)
                input_sequence_length = len(two_sentences)
                line_to_write = {'prompt': two_sentences,'input_seq_len': input_sequence_length,'text_len': text_length}
                output_file.write(json.dumps(line_to_write, ensure_ascii=False) + '\n')

    print("Exported prompts to output file.")
    return

if __name__ == "__main__":
    main()