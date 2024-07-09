# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        v_prompt_tokens: List[int], # only one (existing) prompt, but severaö encoded tokens -> List
        #prompt_tokens: List[List[int]],
        v_next_token: List[int], # only one inside
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        v_bsz = len(v_prompt_tokens) #batch size
        assert v_bsz <= params.max_batch_size, (v_bsz, params.max_batch_size)

        #min_prompt_len = min(len(t) for t in prompt_tokens)
        v_min_prompt_len = len(v_prompt_tokens)

        #max_prompt_len = max(len(t) for t in prompt_tokens)
        v_max_prompt_len = len(v_prompt_tokens)

        assert v_max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + v_max_prompt_len) # either max_seq_len or prompt+max_gen_len

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((v_bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(v_prompt_tokens): 
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")


        #print(tokens, token_logprobs)
        prev_pos = 0

        if v_min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)

        for cur_pos in range(v_min_prompt_len, total_len):
            # array tokens ist als indices-zahlen drin gespeichert
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                # sample_top_p sollte nicht neu token ausgeben, 
                # sondern checken welche WSK für's nächste Token wäre
                #next_token = sample_top_p(probs, top_p)
                v_next_token_prob = v_sample_top_p(probs, top_p, v_next_token[0])
                break # stop loop at once

        #print(v_next_token_prob, "next_token_prob")
        return v_next_token_prob

    def text_completion(
        self,
        prompts: List[str], #just one prompt inside in our case
        v_next_tokens: List[str], # only one continuation sentence inside
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        #print(prompts)
        #print(v_next_tokens)
        v_prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        v_next_tokens_list_of_lists = [self.tokenizer.encode(y, bos=False, eos=False) for y in v_next_tokens]  
        v_next_tokens_list = v_next_tokens_list_of_lists[0]
        v_original_length_next_tokens = len(v_next_tokens_list)
        v_next_tokens_probs = []
        for i in range(v_original_length_next_tokens):
            #print(v_prompt_tokens)
            #print(v_next_tokens_list[:1])
            #vocab = self.tokenizer.n_words
            #print("Vocabulary size:", vocab)
            
            v_next_token_prob = self.generate(
                v_prompt_tokens=v_prompt_tokens,
                v_next_token=v_next_tokens_list[:1], # should be 563
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                echo=echo,
            )
            # Cut off the first element of next_tokens and append it to prompts
            #print(v_prompt_tokens)
            v_prompt_tokens[0].extend([v_next_tokens_list.pop(0)])
            #print(v_prompt_tokens)
            # append the probabilities
            v_next_tokens_probs.append(v_next_token_prob)

        # check if probabilities have same length as the list of "next-tokens"
        all_probs_tensor = torch.cat(v_next_tokens_probs,dim=0)
        #print(all_probs_tensor)
        all_probs_list = all_probs_tensor.tolist()
        #print(all_probs_list)
        probs_list = [item for sublist in all_probs_list for item in sublist]

        assert len(probs_list) == v_original_length_next_tokens, "Lengths of probabilities and next tokens do not match"
        return probs_list


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """

    #print("probls",probs)
    # probs_idx sind index-werte zu probabs
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    #print("sort: probs_sort + probs_idx",probs_sort, probs_sort.size())
    #print("sort: probs_idx # size",[probs_idx, probs_idx.size()])
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    #print("cumsum",probs_sum)
    #mask: difference between probs_sum and probs_sort being greater than p
    mask = probs_sum - probs_sort > p
    #print("mask, > p, p",mask, mask.size(), p)
    probs_sort[mask] = 0.0
    #print("Mask (?)",probs_sort[mask], probs_sort[mask].size())
    #in-place division of the tensor probs_sort by the sum of its elements along the last dimension
    # aka normalization
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    #print("Normalize",probs_sort, probs_sort.size())
    # generates a sample from a multinomial distribution defined by the probabilities in the probs_sort tensor. 
    # It randomly selects one token index based on the probabilities provided in probs_sort. 
    # The num_samples=1 argument specifies that only one sample should be generated. 
    # After this line, next_token contains the index of the selected token.
    next_token = torch.multinomial(probs_sort, num_samples=1) # 
    #print("Multinomial",next_token)
    #print("Do the probs sum up to 1?",torch.sum(probs_sort))
    #print("What was the probability? ")
    # Gather:
    # function to extract the actual token from the probs_idx tensor corresponding to the index selected 
    # in the previous step (next_token). The probs_idx tensor likely contains the indices of
    # the tokens in the original vocabulary. next_token is index for vector with indexes fro probabilities 
    # to corresponding vocab.
    next_token = torch.gather(probs_idx, -1, next_token)
    #print("Gather next token",next_token)

    ## means all in all: get from probs_sort the probability at index 'next token'
    #probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token_prob = torch.gather(probs,-1 ,next_token)
    
    mask = torch.eq(probs_idx, next_token) # find index where val=next token (13)
    # Find indices where mask is True
    indices = torch.nonzero(mask)
    #print("indices",indices)

    mask = torch.eq(probs, next_token_prob) # find index where val=next token (13)
    # Find indices where mask is True
    indices = torch.nonzero(mask)
    #print("indices",indices)

    #print("next_token_prob",next_token_prob)
    #return
    return next_token, next_token_prob

def v_sample_top_p(probs, p, v_next_token_vocab_idx):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    # make tensor of int
    batch_size = 1
    num_tokens = 1
    v_next_token_vocab_idx_tensor = torch.zeros((batch_size, num_tokens), dtype=torch.long, device="cuda")
    v_next_token_vocab_idx_tensor[0, 0] = v_next_token_vocab_idx

    # Gather:
    # function to extract the actual token from the probs_idx tensor corresponding to the index selected 
    # in the previous step (next_token). The probs_idx tensor likely contains the indices of
    # the tokens in the original vocabulary. next_token is index for vector with indexes fro probabilities 
    # to corresponding vocab.
    ## means all in all: get from probs_sort the probability at index 'v_next_token_vocab_idx_tensor'
    next_token_prob = torch.gather(probs,-1 ,v_next_token_vocab_idx_tensor)

    #print("next_token_prob",next_token_prob, "next_token_vocab_idx", v_next_token_vocab_idx_tensor)
    return next_token_prob