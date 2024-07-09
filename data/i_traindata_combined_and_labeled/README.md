# Dataset creation step by step

## Preprocessing of text
Refer to Readme in `data/c_language_proprocessing`


## Determine token length of texts
The script is in `statistics/data_len_in_tokens.ipynb`

Run the script. Here you will retrieve two things: 
- The tokenlength of all texts in a dict, in combination with their indices, per origin (llama, mistral, human written) and per genre (wikipedia, taz, ...). This is stored in statistics/output
- the plots for the kernel density and boxplots about the data distribution in terms of the length of tokens, inclusively their quartiles per origin and genre.


## Pick quartiles
The script is in `data/i_traindata_combined_and_labeled`.

`pick_quartiles_first_step.py`

Here you have to configure the very last fuction call, for generating the dictionaries: The first two position are the defined as `DESIRED_NUM_OF_INDICES_PER_CLASS_HW` and `DESIRED_NUM_OF_INDICES_PER_CLASS_MG`. The third is a bool to decide weather it should be saved or only loaded for further processing.

### Complete Dataset Quartiles
```
if __name__ == "__main__":
    get_indices_per_quartiles(None,None,True)
```
`None` will consider the whole dataset. So for saving, the variable `FILENAME_JSON_INDICES_DICT_HW_COMPLETE_DATASET, FILENAME_JSON_INDICES_DICT_MG_COMPLETE_DATASET_LLAMA` or `FILENAME_JSON_INDICES_DICT_MG_COMPLETE_DATASET_MIXED` have to be chosen in the according function!

### Test Dataset Quartiles
Next, you should do the test-set.
```
if __name__ == "__main__":
    get_indices_per_quartiles(2500,2500,True)
```
For saving, the variable `FILENAME_JSON_INDICES_DICT_HW_TEST_FIX` and either `FILENAME_JSON_INDICES_DICT_MG_TEST_FIX_LLAMA` or `FILENAME_JSON_INDICES_DICT_MG_TEST_FIX_MIXED` should be chosen.

--- 

Next, the "Pool" is generated, which is everything exept of the test-set. This is done by just deleting the indices of the test set from the complete dataset.
This step takes around half an hour.

MAKE a backup of the test-dataset!!!

### Deleting the indices of the test set from the complete dataset
This is done with the script `split_fix_test_dataset_second_step.py`.

Please configure first, if you want to slice the normal human-written and llama dataset, or the mixed one.
```
MIXED_MODELS_ONLY = 0 # or 1
```
Remember, that this script should only be executed once for each dataset, so one time with `0` and one time with `1`. Only repeat, if you changed something in your datasets or added new data.

Then just run it with `python split_fix_test_dataset_second_step.py`.

The new generated files are 
- `'indices_per_genre_and_quartiles_hw_pool_without_test.json'`
- `'indices_per_genre_and_quartiles_mg_pool_without_test_llama.json'` and 
- `'indices_per_genre_and_quartiles_mg_pool_without_test_mixed_models.json'` 
and they can be accessed everywhere with the variables 
- `FILENAME_JSON_INDICES_DICT_HW_TRAIN_EVAL_POOL`,
- `FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_POOL_LLAMA` and 
- `FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_POOL_MIXED`.

### Train-and-Evaluation Dataset
Back to `pick_quartiles_third_step.py`. It will take from the "pool" dataset if you change the source from COMPLETE to POOL. You should do that for this step.
```
if __name__ == "__main__":
    get_indices_per_quartiles(22500,22500,True)
```
FILENAME_JSON_INDICES_DICT_HW_TRAIN_EVAL => 'indices_per_genre_and_quartiles_hw_train_eval.json'
FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_LLAMA => 'indices_per_genre_and_quartiles_mg_train_eval_llama.json'
FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_MIXED = >'indices_per_genre_and_quartiles_mg_train_eval_mixed_models.json'

### Available dictionaries in `statistics/output`:

- `'tokenlength_dict_hw.json'`
- `'tokenlength_dict_mg_llama.json'`
- `'tokenlength_dict_mg_mixed_models.json'`

- `indices_per_genre_and_quartiles_hw_complete_dataset.json`. This is the quartile indices distribution for all human written data
- `'indices_per_genre_and_quartiles_mg_complete_dataset_llama.json'`. This is the quartile indices distribution for all llama generated data.
- `'indices_per_genre_and_quartiles_mg_complete_dataset_mixed_models.json'`. This is the quartile indices distribution for all mixed models' generated data.
- `indices_per_genre_and_quartiles_hw_test_fix`. This is the human written quartile indices distribution for the test dataset
- `indices_per_genre_and_quartiles_mg_test_fix_llama`. This is the llama generated quartile indices distribution for the test dataset
- `indices_per_genre_and_quartiles_mg_test_fix_mixed.json`. This is the mixed models' generated quartile indices distribution for the test dataset
- `indices_per_genre_and_quartiles_hw_pool_without_test.json`. This is the human written quartile indices distribution for the rest dataset
- `indices_per_genre_and_quartiles_mg_pool_without_test_llama.json`. This is the llama generated quartile indices distribution for the rest dataset
- `indices_per_genre_and_quartiles_mg_pool_without_test_mixed.json`. This is the mixed models' generated quartile indices distribution for the rest dataset

- 'indices_per_genre_and_quartiles_hw_train_eval.json'
- 'indices_per_genre_and_quartiles_mg_train_eval_llama.json'
- 'indices_per_genre_and_quartiles_mg_train_eval_mixed_models.json'

- the test backup files (3)

- the plotted tokenlength distributions (4)



## 2 Steps: Text dataset for easy reload
`combine_dataset_from_indices_forth_step.py`

This takes a while, so better save it. You need to configure if Test or Train!
at four places.
Either
`TRAIN_EVAL` or `TEST`

FILENAME_JSON_INDICES_DICT_HW_TRAIN_EVAL => FILENAME_JSON_INDICES_DICT_HW_TEST_FIX

FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_LLAMA => FILENAME_JSON_INDICES_DICT_MG_TEST_FIX_LLAMA

FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_MIXED => FILENAME_JSON_INDICES_DICT_MG_TEST_FIX_MIXED

FILENAME_HW_DATASET_TEXTS_TRAIN_EVAL => FILENAME_HW_DATASET_TEXTS_TEST

FILENAME_MG_DATASET_TEXTS_TRAIN_EVAL_LLAMA => FILENAME_MG_DATASET_TEXTS_TEST_LLAMA

FILENAME_MG_DATASET_TEXTS_TRAIN_EVAL_MIXED => FILENAME_MG_DATASET_TEXTS_TEST_MIXED



```
train_eval_indices_per_quartile_dict_hw = load_json(FILENAME_JSON_INDICES_DICT_HW_TRAIN_EVAL)
    train_eval_indices_per_quartile_dict_mg_llama = load_json(FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_LLAMA)

    # question: do the entries have to be mixed by label? -> No, it is done in the DataLoader
    texts=[]
    labels=[]

    hw_texts, hw_labels = load_human_written_dataset(train_eval_indices_per_quartile_dict_hw)
    mg_texts, mg_labels = load_machine_generated_dataset_llama(train_eval_indices_per_quartile_dict_mg_llama)

    texts = hw_texts + mg_texts
    labels = hw_labels + mg_labels

    hw_path_object = os.path.join(OS_PATH_DATASET_TEXTS_ONLY,FILENAME_HW_DATASET_TEXTS_TRAIN_EVAL)
    with open(hw_path_object, 'w', encoding='utf-8') as out_hw:
        for entry in hw_texts:
            json.dump(entry, out_hw,ensure_ascii=False)  # Write the JSON object
            out_hw.write('\n')        # Add a newline character

    mg_path_object = os.path.join(OS_PATH_DATASET_TEXTS_ONLY,FILENAME_MG_DATASET_TEXTS_TRAIN_EVAL_LLAMA)
```
## Reload it