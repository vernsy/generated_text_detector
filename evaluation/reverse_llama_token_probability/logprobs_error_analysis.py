# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import nltk
import random

from llama_reverse import Llama
from typing import List
import argparse
import json

# for cleaning cache before running
import torch
import gc
import os, sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=32'
# make it possible to find the data python module in parent folder
#sys.path.insert(0, '..') 
#import setup.config as config

#INPUT_PATH = config.PROJECT_PATH + config.OS_PATH_ERROR_ANALYSIS
#INPUT_FILE = config.FILENAME_MIXED_TEXTS_TRUE_AND_FALSE_POSITIVES_NEGATIVES
INPUT_FILE = '../j_data_for_error_analysis/dataset_for_error_analysis.tsv'

GERMAN_AVERAGE_SENTENCE_LENGTH = 25
FAILED_AT_LINE = 4640 #this is in case the program stops, enter line number logged in the logging file
# wikipedia: skipped line 3331, bc too long prompt, delete later also in input files!
def cut_at_character_n(text, num_characters):
    # randomize if generated text is longer or shorter than original
    randomizer = random.choice([-1, 1])*GERMAN_AVERAGE_SENTENCE_LENGTH
    cutted_text = text[:num_characters+randomizer] # if longer than text, python can handle it
    return cutted_text

def cut_at_last_complete_sentence(text):
    sentences = nltk.sent_tokenize(text)
    cut_text = ' '.join(sentences[:-1])
    return cut_text

def log_probabilities_to_file(output_file_path,tokens,logprobs) -> None:
    for tok in tokens:
        for i,prob in enumerate(logprobs):
            if not i == 0:
                line1 += '\t'
                line2 += '\t'
            line1 += tok
    return

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4, # before was 4, reduced because of long texts of wikimedia
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 

    prompt_path = args.prompt_path
    output_file = args.out_path
    logging_path = args.logging_path
    print(f'input file: {INPUT_FILE}')

    # calculate maximal sequence length of the input once
    max_seq_len = 0
    convert_to_int_or_zero = lambda value: int(value) if str(value).isdigit() else 0


    #set for wikimedia and wikipedia, bc some seem to be very long, too long for cuda memory handling
    max_seq_len = 2000
    
    print(f"Maximal sequence length for the generator input is {max_seq_len}.")

    # build generator once
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    line_index = 0
    #input_path = os.path.join(INPUT_PATH, INPUT_FILE)
    with open(INPUT_FILE, 'r') as file:
        for line in file:
            if line_index == 0:
                line_index += 1
                continue #skip row 0 
            line = line.strip()

            # Split the line based on tab ('\t') delimiter
            columns = line.split('\t')

            # Split the first two sentences as a prompt
            prompt = ' '.join(columns[0].split('.')[:2])
            predicted_label = columns[1]
            true_label = columns[2]

            # The rest as text
            text_continuation = ' '.join(columns[0].split('.')[2:]) if len(columns[0].split('.')) > 2 else ""
            print(prompt)
            print(text_continuation)
            if len(prompt) == 0 or len(text_continuation)==0:
                print(line_index)
                continue
            max_gen_len = len(text_continuation)+len(prompt)
            prompts: List[str] = [prompt]
            v_next_tokens: List[str] = [text_continuation]

            # iterate through each prompt, because we want to adopt "max_gen_len" 
            # values, dependant on the prompt input
            results = generator.text_completion(
                prompts,
                v_next_tokens=v_next_tokens,
                max_gen_len=max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p,
                logprobs=True
            )
            results += predicted_label
            results += true_label
            print("result in 03_logprobs_error_analysis",results)
            tsv_string = '\t'.join(map(str, results))
            with open(output_file, 'a+') as f_out: # a+ will append
                f_out.write(tsv_string + '\n')
            line_index += 1
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--local-rank', type=int, required=False, default=0)
    parser.add_argument('--max_batch_size', type=int, required=False, default=16) # changed for long texts of wikimedia, before 32
    parser.add_argument('--top_p', type=float, required=False, default=0.9)
    parser.add_argument('--temperature', type=float, required=False, default=0.8)
    parser.add_argument('--max_seq_len', type=int, required=False, default=512)
    parser.add_argument('--max_gen_len', type=int, required=False, default=512)
    parser.add_argument('--prompt_path', type=str, required=False, default='./prompt.txt')  # path to prompt file
    parser.add_argument('--out_path', type=str, required=False, default='./generated.txt') # path to generated file
    parser.add_argument('--logging_path', type=str, required=False, default='./lastline.log') # path to log file
    parser.add_argument('--nproc_per_node', type=int, required=False, default=2) # For 13B MP value is 2
    
    args = parser.parse_args()
    print(args)
    
    print(f'Checkpoint set to {args.ckpt_dir}')
    print(f'Tokenizer set to {args.tokenizer_path}')
    fire.Fire(main) # arguments inputted from the command line will overwrite
