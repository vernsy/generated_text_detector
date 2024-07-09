# Setup and Installation

## Step-by-step Guide

:exclamation: For running the project, you need to change the name of `config.ini.copy` to `config.ini` before you fill in your credentials :exclamation:

This is only necessary in case you want to run scripts that involve webresources like APIs and need a key or password.
The `config.ini` is listed in `.gitignore` and will not be loaded into the git-repository
"""

### 1: Fill config.ini

In the `config.ini` you have to list your credentials (API-Keys) and the local paths to your data. 

### 2: Do NOT change the config.py

This file contains many internal paths to datasets and input/output files of scripts. Unless you don't explicitely want to have the data elsewhere, you shouldn't change anything here :exclamation:

### 3: Installation

Create your own virtual environment, either with `pip` or with `conda`, whatever you prefer. Then enter this environment.
Go into the setup folder and install all requirements. 
```
cd setup
pip install -r requirements.txt
```
Please run manually: 
```
python -m spacy download de_core_news_lg
apt install python3-tk # for module tkinter
```
After install requirements: Run `wandb login` and enter your api key in the terminal, in case you want to use the weights-and-biases web-logging-tool that creates really nice evaluation plots and provides easy calculations of deviations, optimums etc.

### Next step: proceed in the data subfolder
You can refer to the `README.md` there.