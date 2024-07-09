## General information

### Code

We have a user called llama.
The code (scripts to use the LLMs) can be found in `/home/llama/git...` (which is a git-repository).

### Models

They can be found in  `/data/LLaMA/<X>B/` .
They are called "7B", "13B", "30B" and "65B". (7B means seven billion parameters. In total there are 4 models of different sizes.)
This directory is named later in the `torchrun` command the 'Checkpoint Directory'. This is due to the fact that during training, the intermediate states of the model have been saved there. In the call `load_lm()`, the program run by `torchrun` will load (the latest) checkpoint. See below for more explanation on the [torchrun script](#the-torchrun-script).


## Preparation

For the python virtual environment use miniconda-tool. Miniconda can be installed like this:

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
conda list
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```
To work within your own environment with the model, you should install a virtual environment from which you can interact with the model(s).
```
conda list # lists all available environments
cd <your_working_directory>
conda create -n llama # llama is the name, but could be anything
conda env list
conda activate llama
```

Virtual environment is started with `conda activate llama`.

## Running the model

Install the necessary tools via miniconda and for the correct GPU. In your home directory create your own folder, e.g. `mkdir -p masterthesis/data_generation`

Then install...
- cuda (the driver for your Nvidia GPU)
- pytorch, torchvision and torchaudio (pytorch to use torchvision and torchaudio with the pytorch library), then compile pytorch for this specific versions and for this specific GPU
- git
- cudatoolkit for usage of cuda


```
conda install cuda -c nvidia/label/cuda-11.6.0 -c nvidia/label/cuda-11.6.1 -c nvidia/label/cuda-11.6.2
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install git # probably no version conflicts, seems to be just additional need for git
conda install cudatoolkit=11.3.1 # 11.6 is not available, only 11.8 and that is probably too new


# The old command below does not work anymode, because cuda 11.6 needs matching versions of pytorch and 
# they are not the newest anymore, so we need to spacify them explicitly
# conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge --solver=libmamba 
# conda-forge is an additional channel from which packages could be installed
# the solver libmamba is in case you get resolve errors due to package website downtimes. it is a different solver
# https://stackoverflow.com/questions/74781771/how-we-can-resolve-solving-environment-failed-with-initial-frozen-solve-retry
```
After that, install the requirements.
```
cd ~/git/llama/ # this should have been copied to your home directory
cp requirements.txt ../../masterthesis/data_generation/
cd
cd masterthesis/data_generation/
pip install -r requirements.txt pip install -e .
cd ../../git/llama/
cat example13B.py # to catch the steps that are performed to interact with the model
```


```
python -m torch.distributed.launch example13B.py --max_batch_size=1
```

### Installation Troubleshooting

For debugging use:
```
python -m torch.utils.collect_env
```
Make sure that you have installed the GPU packages, not the CPU ones. For "Cuda available" value you should see `True`. Cuda is the "driver" that can communicate with Nvidia GPUs for parallelization, the special feature we want from PyTorch. For help about NCCL: https://discuss.pytorch.org/t/runtimeerror-distributed-package-doesnt-have-nccl-built-in/176744/8


Make sure that you GPU driver is installed and not too old for your pytorch and cuda versions, because otherwise it will install the CPU binaries.
https://discuss.pytorch.org/t/pytorch-not-getting-compiled-with-gpu-when-using-conda-install/174655/5

- For driver installation for nvidia see:
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements
- Cuda compatibility with pythorch and torch packages for NVIDIA A6000 can be found here:
https://de.wikipedia.org/wiki/CUDA
NVIDIA A6000 is the model version number of our NVIDIA GPU on nofake. For different GPUs you might need completely different versions. Keep in mind, that with other versions your python scripts might throuw errors.
- If your GPU has only compatibility with older cuda versions, check this website for the pytorch / torchvision /torchaudio, that are matching with this cuda version (click on 'install previous versions') https://pytorch.org/get-started/locally/ .
- Maybe helpful for debugging: https://stackoverflow.com/questions/76376486/how-to-install-pytorch-with-cuda-support-using-conda


## The torchrun script

In the script, the model is loaded, called the `generator`, because in our case it is used to generate texts. We can define, which model we want to choose by determining the desired parameter number via the `--chkpt-dir` parameter and the checkpoint intself by the `--local-rank` parameter. This way of choosing the checkpoint only gets logical, if you consider the script as a modified training script, where you choose the (multiple) checkpoints by the ranking they have to be trained - in that case, not in our's. 

### Worldsize and NCCL (Parallelization)

NCCL (Nvidia Collective Communication Library) is a module/backend for communication between GPU devices. We need it for a distributed setup over 2 GPUs.
After initializing, we use the builtin function `initialize_model_parallel(world_size)` from the python module `fairscale.nn.model_parallel.initialize`. We use it in our case for model parallelization, when a single model doesn't fit or could work quicker when we apply it just into the memory of a single GPU. The model's parameters are then distributed across the 2 GPUs. The `nn` stands for neural network.

The world size should be equal to the number of GPUs (on the local machine, or if available, then even on a distributed setup, "the world"). In our case, 2, because we only have our signle noFake server. The world size cannot be passed to our script via flag, only by environment variable.


```python
    world_size = int(os.environ.get("WORLD_SIZE", -1)) # default is -1, which disables parallelization

    ...

    torch.distributed.init_process_group("nccl")       # load nccl
    initialize_model_parallel(world_size)              # number of parallel processes /distributed parts of the model
                                                       # this apparently performs some kind of fork() (like in C language)
                                                       # where the child processes continue from this line on as singe process

    torch.cuda.set_device(local_rank)                  # this and above command only interesting when training; but /TODO check on huggingface if those two checkpoints can also be used in parallel when generating text. This function is a no-op if the argument is negative.

    # seed must be the same in all processes
    torch.manual_seed(1)
```
It currently appears, that we don't really need the above code.


### Checkpoint Directory and Local rank

The checkpoints of the trained model ly in the checkpoint directory. If we would like to use the 13 Million model, we run the command `torchrun [...] --ckpt_dir /data/LLaMA/13B`.`
```python
    local_rank = int(os.environ.get("LOCAL_RANK", -1))  # default is -1

    ...

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))   # sort the checkpoints
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}" # MP means model parallelism 
    ckpt_path = checkpoints[local_rank]                  # if -1, we choose the newest and most trained model checkpoint
```

### Number of processes per node

--nproc_per_node 2 equals to local world size (workers per node). In our script, this parameter is never used. Anyways both GPUs are used.
`–nproc_per_node` should be set to the MP value for the model you are using.
Different models require different model-parallel (MP) values:

| Model | MP   |
| ----- | ---- |
| 7B    |    1 |
| 13B   |    2 |
| 70B   |    8 |



All models support sequence length up to 4096 tokens, but we pre-allocate the cache according to max_seq_len and max_batch_size values. So set those according to your hardware.

 model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

### Tokenizer
 --tokenizer_path /data/LLaMA/tokenizer.model This is a pretrained LLama Tokenizer. We could also take our own, already tokenized input to save some computing time and resources, but as our tokenizer has been a python spacy tokenizer and not the tokenizer trained by and for LLaMA, we take the tokenizer `/data/LLaMA/tokenizer.model` to prepare our input data best for the generation of text with the LLaMA model. 
             prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.

### Max batch size 
 --max_batch_size=1 
  The maximum batch size for generating sequences. Maybe 4 is a better value, not one by one.
 
 ### Promt path
 --prompt_path='./prompt.txt'


 ### what is top_p

 Top-p probability threshold for nucleus sampling. Defaults to 0.9. nucleus sampling to produce text with controlled randomness.

 ### Hyperparameters

 Are in `/data/LLaMA/<X>B/params.json`.

 {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": -1}
related to a transformer-based language model
dim: dimensionality of the model's hidden states or embeddings (transformer language model: size of the hidden layer.)
multiple_of: the model's dimensionality (dim) should be a multiple of the specified value. It's a constraint on the model's architecture to ensure certain mathematical properties or optimizations.
n_heads: number of attention heads in the multi-head self-attention mechanism. Each attention head attends to different parts of the input sequence, allowing the model to capture different aspects of the relationships between words.
n_layers: The number of layers (or blocks) in the transformer architecture. Each layer typically consists of a multi-head self-attention mechanism followed by feedforward neural networks. Increasing the number of layers can enhance the model's capacity to capture complex patterns in the data.
norm_eps: The epsilon value used in layer normalization, helping stabilize training and improve generalization of the input.
vocab_size: This is the size of the vocabulary, representing the total number of unique tokens in the input data. The model's embedding layer will have vocab_size neurons, each representing a unique token.

https://github.com/facebookresearch/llama/blob/main/llama/generation.py

## How to run the script:

```
torchrun --nproc_per_node 2 02_text_completion_modified.py --ckpt_dir /data/LLaMA/13B/ --tokenizer_path /data/LLaMA/tokenizer.model --prompt_path=prompts/prompts_zeitonline.jsonl --out_path=llama_generated_from_zeitonline.jsonl --logging_path=lastline_zeitonline.log &

```