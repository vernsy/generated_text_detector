# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import nltk
import random

from llama import Llama
from typing import List
import argparse
import json

# for cleaning cache before running
import torch
import gc
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=32'

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
    print(f'input file: {prompt_path}')

    # calculate maximal sequence length of the input once
    max_seq_len = 0
    convert_to_int_or_zero = lambda value: int(value) if str(value).isdigit() else 0

    with open(prompt_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            try:
                json_data = json.loads(line)
                curr_seq_len_str = json_data["input_seq_len"]
                curr_seq_len = convert_to_int_or_zero(curr_seq_len_str)
                if curr_seq_len > max_seq_len:
                    max_seq_len = curr_seq_len
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")

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
    
    with open(prompt_path, 'r', encoding='utf-8') as input_file:

        line_index = 0
        for line in input_file:
            if line_index <= FAILED_AT_LINE -1:
                line_index += 1
                continue
            with open(logging_path,'w') as f_log:
                f_log.write(str(line_index) + '\n') # catch progress

            try:
                json_data = json.loads(line)
                prompt = json_data["prompt"]
                max_gen_len_str = json_data["text_len"] # should be quite es much as the corresponding text
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")

            max_gen_len = convert_to_int_or_zero(max_gen_len_str)
            prompts: List[str] = [prompt]

            # iterate through each prompt, because we want to adopt "max_gen_len" 
            # values, dependant on the prompt input
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p,
            )          

            result_dict = results[0] # always just one result

            # clean the generated text
            result_raw = next(iter(result_dict.values()))
            result_one_line = result_raw.replace('\n', ' ')
            result_stripped = result_one_line.strip()
            cutted_result = cut_at_character_n(result_stripped,max_gen_len)
            result_complete_sentences = cut_at_last_complete_sentence(cutted_result)

            text_len = len(result_complete_sentences)

            with open(output_file, 'a+') as f_out: # a+ will append
                line_to_write = {"prompt": prompt, "llama_text": result_complete_sentences, "gen_text_len": text_len, "orig_text_len": max_gen_len_str}
                f_out.write(json.dumps(line_to_write, ensure_ascii=False) + '\n')

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
