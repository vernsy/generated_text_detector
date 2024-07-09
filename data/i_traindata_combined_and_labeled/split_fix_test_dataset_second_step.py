import os
import json

from pick_quartiles_first_step import save_dicts_updated_indices_per_genre_and_quartile
from util_i import *

MIXED_MODELS_ONLY = 1

def delete_test_indices_from_normal_dataset_pool(full_dict,test_dict):
    # Iterate over keys in complete_dataset_dict
    for key, nested_dict in full_dict.items():
        # Check if the key exists in test_dataset_dict
        if key in test_dict:
            # Iterate over nested keys in complete_dataset_dict
            for nested_key, listA in nested_dict.items():
                # Check if the nested key exists in test_dataset_dict and if yes, remove it
                if nested_key in test_dict[key]:
                    listB = test_dict[key][nested_key]
                    print(len(listA)," - ",len(listB), " =")
                    full_dict[key][nested_key] = [x for x in listA if x not in listB]
                    print(len(full_dict[key][nested_key]))
    return full_dict


def load_json(FILENAME):
    path_object = os.path.join(OS_PATH_STATISTICS_OUTPUT,FILENAME)
    with open(path_object, 'r') as file:
        indices_per_quartile_dict = json.load(file)
    return indices_per_quartile_dict


def main():

    if not MIXED_MODELS_ONLY:
        # load jsons
        full_indices_per_quartile_dict_hw = load_json(FILENAME_JSON_INDICES_DICT_HW_COMPLETE_DATASET)
        test_indices_per_quartile_dict_hw = load_json(FILENAME_JSON_INDICES_DICT_HW_TEST_FIX)
        full_indices_per_quartile_dict_mg_llama = load_json(FILENAME_JSON_INDICES_DICT_MG_COMPLETE_DATASET_LLAMA)
        test_indices_per_quartile_dict_mg_llama = load_json(FILENAME_JSON_INDICES_DICT_MG_TEST_FIX_LLAMA)

        shortened_dict_hw = delete_test_indices_from_normal_dataset_pool(full_indices_per_quartile_dict_hw,
                                                                        test_indices_per_quartile_dict_hw)
        shortened_dict_mg_llama = delete_test_indices_from_normal_dataset_pool(full_indices_per_quartile_dict_mg_llama,
                                                                        test_indices_per_quartile_dict_mg_llama)
    
        save_dicts_updated_indices_per_genre_and_quartile(FILENAME_JSON_INDICES_DICT_HW_TRAIN_EVAL_POOL, shortened_dict_hw)
        save_dicts_updated_indices_per_genre_and_quartile(FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_POOL_LLAMA, shortened_dict_mg_llama)

    else: 
        # load jsons
        full_indices_per_quartile_dict_mg_mixed = load_json(FILENAME_JSON_INDICES_DICT_MG_COMPLETE_DATASET_MIXED)
        test_indices_per_quartile_dict_mg_mixed = load_json(FILENAME_JSON_INDICES_DICT_MG_TEST_FIX_MIXED)

        shortened_dict_mg_mixed = delete_test_indices_from_normal_dataset_pool(full_indices_per_quartile_dict_mg_mixed,
                                                                        test_indices_per_quartile_dict_mg_mixed)
    
        save_dicts_updated_indices_per_genre_and_quartile(FILENAME_JSON_INDICES_DICT_MG_TRAIN_EVAL_POOL_MIXED, 
                                                          shortened_dict_mg_mixed)


    return

main()