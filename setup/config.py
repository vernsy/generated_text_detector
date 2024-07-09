import configparser
import os
import sys
import setup.config

config_parser = configparser.ConfigParser()

# be able to find config.ini
config_ini_dir = os.path.dirname(os.path.abspath(__file__))
config_ini_file_path = os.path.join(config_ini_dir, 'config.ini')

config_parser.read(config_ini_file_path)

#print(config_parser.sections()) 

""" For running the project, you need to change the name of 
    config.ini.copy to config.ini and fill in your credentials
    there. This is only necessary in case you want to run a 
    script on a certain API and need a key or password.
    The config.ini is listed in .gitignore and will not be 
    loaded into the git-repository
"""

# Global variables
PROJECT_PATH = config_parser.get("Personal","PATH_TO_PROJECT")
MODEL_PATH = config_parser.get("Personal","PATH_TO_MODELS")
DATA_PATH = config_parser.get("Personal","PATH_TO_DATA")
GPT_API_KEY = config_parser.get("Credentials","OPENAI_API_KEY")


# data
## machine_generated llama
FILENAME_MACHINE_GENERATED_LLAMA_SPRINGERNATURE = 'llama_generated_from_springernature_new_no_duplicates.jsonl'
FILENAME_MACHINE_GENERATED_LLAMA_TAZ = 'llama_generated_from_taz_new_no_duplicates.jsonl'
FILENAME_MACHINE_GENERATED_LLAMA_WIKIMEDIA = 'llama_generated_from_wikimedia_new_no_duplicates.jsonl'
FILENAME_MACHINE_GENERATED_LLAMA_WIKIPEDIA = 'llama_generated_from_wikipedia_new_no_duplicates.jsonl'
FILENAME_MACHINE_GENERATED_LLAMA_ZEITONLINE = 'llama_generated_from_zeitonline_new_no_duplicates.jsonl'
OS_PATH_MACHINE_GENERATED_TRAININGDATA_LLAMA = '' # currently all in root

## machine_generated wizard train_eval
OS_PATH_MG_DATA_WIZARD_SNOOZY_MISTRAL_GPT = '/data/h_traindata_machine_generated'

FILENAME_MG_DATA_WIZARD_SPRINGERNATURE = 'wizard_generated_springernature.tsv'
FILENAME_MG_DATA_WIZARD_TAZ = 'wizard_generated_taz.tsv'
FILENAME_MG_DATA_WIZARD_WIKIPEDIA = 'wizard_generated_wikipedia.tsv'
FILENAME_MG_DATA_WIZARD_ZEITONLINE = 'wizard_generated_zeitonline.tsv'

## machine_generated snoozy train_eval
FILENAME_MG_DATA_SNOOZY_SPRINGERNATURE = 'snoozy_generated_springernature.tsv'
FILENAME_MG_DATA_SNOOZY_TAZ = 'snoozy_generated_taz.tsv'
FILENAME_MG_DATA_SNOOZY_WIKIPEDIA = 'snoozy_generated_wikipedia.tsv'
FILENAME_MG_DATA_SNOOZY_ZEITONLINE = 'snoozy_generated_zeitonline.tsv'

## machine_generated mistral train_eval
FILENAME_MG_DATA_MISTRAL_TAZ = 'mistral_leo_generated_taz.tsv'

## machine_generated gpt train_eval
FILENAME_MG_DATA_GPT_SPRINGERNATURE = 'gpt_generated_springernature.tsv'
FILENAME_MG_DATA_GPT_TAZ = 'gpt_generated_taz.tsv'
FILENAME_MG_DATA_GPT_WIKIPEDIA = 'gpt_generated_wikipedia.tsv'
FILENAME_MG_DATA_GPT_ZEITONLINE = 'gpt_generated_zeitonline.tsv'

## machine_generated test 
OS_PATH_MG_DATA_TEST_MISTRAL_GPT = '/data/g_testdata_machine_generated'

FILENAME_MG_DATA_GPT_WIKIMEDIA = 'gpt_generated_wikimedia.tsv'
FILENAME_MG_DATA_MISTRAL_WIKIPEDIA = 'mistral_leo_generated_wikipedia.tsv'

## human written
### raw
OS_PATH_HUMAN_WRITTEN_RAW = '/data/a_raw_and_unprocessed_data_downloads'
FILENAME_HUMAN_WRITTEN_RAW_ZEITONLINE = 'zeitonline.jsonl'

### scraped
OS_PATH_HUMAN_WRITTEN_SCRAPED = '/data/b_scrapers_for_human_written_text/output_spiders'
FILENAME_HUMAN_WRITTEN_SCRAPED_SPRINGERNATURE = 'output_springernature.json'
FILENAME_HUMAN_WRITTEN_SCRAPED_TAZ = 'output_taz.jsonl'
FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIMEDIA = 'output_wikimedia.jsonl'
FILENAME_HUMAN_WRITTEN_SCRAPED_WIKIPEDIA = 'output_wikipedia.jsonl'

### cleaned
OS_PATH_HUMAN_WRITTEN_CLEANED = '/data/c_language_preprocessing/output_cleaned_data'
FILENAME_HUMAN_WRITTEN_CLEANED_SPRINGERNATURE = 'output_springernature.jsonl'
FILENAME_HUMAN_WRITTEN_CLEANED_TAZ='output_taz.json'
FILENAME_HUMAN_WRITTEN_CLEANED_WIKIMEDIA = 'output_wikimedia.jsonl'
FILENAME_HUMAN_WRITTEN_CLEANED_WIKIPEDIA = 'output_wikipedia.jsonl'
FILENAME_HUMAN_WRITTEN_CLEANED_ZEITONLINE = 'output_zeitonline.jsonl'

### prompts
OS_PATH_HUMAN_WRITTEN_PROMPTS = '/data/e_llama_text_generation/prompts'
FILENAME_HUMAN_WRITTEN_PROMPTS_SPRINGERNATURE = 'prompts_springernature.jsonl'
FILENAME_HUMAN_WRITTEN_PROMPTS_TAZ = 'prompts_taz.jsonl'
FILENAME_HUMAN_WRITTEN_PROMPTS_WIKIMEDIA = 'prompts_wikimedia.jsonl'
FILENAME_HUMAN_WRITTEN_PROMPTS_WIKIPEDIA = 'prompts_wikipedia.jsonl'
FILENAME_HUMAN_WRITTEN_PROMPTS_ZEITONLINE = 'prompts_zeitonline.jsonl'

### text only datasets
OS_PATH_DATASET_TEXTS_ONLY = '/data/i_traindata_combined_and_labeled/output'
FILENAME_HW_DATASET_TEXTS_TRAIN_EVAL = 'hw_train_eval_texts.jsonl'
FILENAME_MG_DATASET_TEXTS_TRAIN_EVAL_LLAMA = 'mg_train_eval_texts_llama.jsonl'
FILENAME_MG_DATASET_TEXTS_TRAIN_EVAL_MIXED = 'mg_train_eval_texts_mixed_models.jsonl'

FILENAME_HW_DATASET_TEXTS_TEST = 'hw_test_texts.jsonl'
FILENAME_MG_DATASET_TEXTS_TEST_LLAMA = 'mg_test_texts_llama.jsonl'
FILENAME_MG_DATASET_TEXTS_TEST_MIXED = 'mg_test_texts_mixed_models.jsonl'

OS_PATH_DATASET_TEXTS_ONLY_TEST_SPECIAL = '/data/g_testdata_machine_generated'
FILENAME_MG_DATASET_TEXTS_TEST_SPECIAL_GPT = 'gpt_generated_wikimedia.tsv'
FILENAME_MG_DATASET_TEXTS_TEST_SPECIAL_MISTRAL = 'mistral_leo_generated_wikipedia.tsv'

# model
OS_PATH_TO_FINETUNED_MODEL_MIXED = MODEL_PATH + '/llama-trained-model'
OS_PATH_TO_FINETUNED_MODEL_LLAMA = MODEL_PATH + '/mixed-trained-model'
#evaluation
OS_PATH_LOGGING_PERFORMANCE_METRICS = '/evaluation/logs'
OUTPUT_LOGGING_FILE_PERFORMANCE_METRICS_BASELINE = 'performance_metrics_baseline_model.tsv'

## error analysis
OS_PATH_ERROR_ANALYSIS_TEXTS = '/data/j_data_for_error_analysis'
FILENAME_MIXED_TEXTS_TRUE_AND_FALSE_POSITIVES_NEGATIVES = 'dataset_for_error_analysis.tsv'
OS_PATH_ERROR_ANALYSIS_PROBABILITIES = '/evaluation/reverse_llama_token_probability'
FILENAME_PROBABILITIES_ERROR_ANALYSIS = 'error_analysis.tsv'

# statistics
OS_PATH_STATISTICS_OUTPUT = '/statistics/output'
FILENAME_TOKENLENGTH_DICT_HW = 'tokenlength_dict_hw.json'
FILENAME_TOKENLENGTH_DICT_MG_LLAMA = 'tokenlength_dict_mg_llama.json'
FILENAME_TOKENLENGTH_DICT_MG_MIXED = 'tokenlength_dict_mg_mixed_models.json'
FILENAME_TOKENLENGTH_DICT_MG_TEST_ONLY = 'tokenlength_dict_mg_test_only.json'

FILENAME_JSON_INDICES_DICT_HW_TRAIN_EVAL = 'indices_per_genre_and_quartiles_hw_train_eval.json'
FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_LLAMA = 'indices_per_genre_and_quartiles_mg_train_eval_llama.json'
FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_MIXED = 'indices_per_genre_and_quartiles_mg_train_eval_mixed_models.json'  # same amount as pool, so just copied

FILENAME_JSON_INDICES_DICT_HW_TRAIN_EVAL_POOL = 'indices_per_genre_and_quartiles_hw_pool_without_test.json'
FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_POOL_LLAMA = 'indices_per_genre_and_quartiles_mg_pool_without_test_llama.json'
FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_POOL_MIXED = 'indices_per_genre_and_quartiles_mg_pool_without_test_mixed_models.json'

FILENAME_JSON_INDICES_DICT_HW_COMPLETE_DATASET = 'indices_per_genre_and_quartiles_hw_complete_dataset.json'
FILENAME_JSON_INDICES_DICT_MG_COMPLETE_DATASET_LLAMA = 'indices_per_genre_and_quartiles_mg_complete_dataset_llama.json'
FILENAME_JSON_INDICES_DICT_MG_COMPLETE_DATASET_MIXED = 'indices_per_genre_and_quartiles_mg_complete_dataset_mixed_models.json'
FILENAME_JSON_INDICES_DICT_MG_COMPLETE_DATASET_MIXED_TEST_ONLY = 'indices_per_genre_and_quartiles_mg_complete_dataset_mixed_test_only.json'

FILENAME_JSON_INDICES_DICT_HW_TEST_FIX = 'indices_per_genre_and_quartiles_hw_test_fix.json'
FILENAME_JSON_INDICES_DICT_MG_TEST_FIX_LLAMA = 'indices_per_genre_and_quartiles_mg_test_fix_llama.json'
FILENAME_JSON_INDICES_DICT_MG_TEST_FIX_MIXED = 'indices_per_genre_and_quartiles_mg_test_fix_mixed_models.json'