import json
import sys, os
from typing import Union, Dict,List
import numpy as np
import random
# make it possible to find the data python module in parent folder
sys.path.insert(0, '../..') 
from data.i_traindata_combined_and_labeled.util_i import *


### load all dicts
# load jsons
def load_json_tokenlengths(FILENAME: str) -> Dict[str, Dict[str, int]]:
    path_object = os.path.join(
        OS_PATH_STATISTICS_OUTPUT,
        FILENAME
    )
    with open(path_object, 'r') as in_file:
        token_lens_dict = json.load(in_file)
        return token_lens_dict
    
### convert dict of dicts to dict of lists
def get_dict_with_lists(token_lens_dict):
    new_dict_with_lists = {}
    for filename_key, key_value_dict in token_lens_dict.items():
        ## count all values without their keys to a list
        values = []
        for _key,value in key_value_dict.items():
            values += [value]
        new_dict_with_lists[filename_key] = values
    return new_dict_with_lists

### get quartiles
def get_quartiles(token_lens_dict) -> Dict[str,Union[int, int, int]]:
    new_dict_with_lists = get_dict_with_lists(token_lens_dict)
    dict_with_quartiles_per_genre = {}
    for key, values in new_dict_with_lists.items():
        # Ensure values are lists
        if not isinstance(values, list):
            values = [values]
        # Convert text lengths to integers
        token_lens_int = [int(token_len) for token_len in values]
        # Calculate quartiles
        token_len_quartiles = np.percentile(token_lens_int, [25, 50, 75])
        dict_with_quartiles_per_genre[key] = token_len_quartiles
    print(dict_with_quartiles_per_genre)
    return dict_with_quartiles_per_genre

### divide into lists of indices per quartile
def sort_indices_per_genre_and_quartile(token_lens_dict: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, List[int]]]:
    # thresholds dict
    thresholds_of_quartiles_per_genre_dict = get_quartiles(token_lens_dict)
    return_dict_with_nested_dict_quartile_lists = {}
    for key,index_value_dict in token_lens_dict.items():
        q0 = []
        q1 = []
        q2 = []
        q3 = []
        # thresholds
        t1,t2,t3 = thresholds_of_quartiles_per_genre_dict[key]
        print("key:",key,"thresholds:", t1,t2,t3)
        return_dict_with_nested_dict_quartile_lists[key] = {}
        temp_dict_with_quartile_lists = {}
        for index,value in index_value_dict.items(): # attention, no enumerate because not all indices are present
            if value < t1:
                q0 += [index]
            elif value < t2:
                q1 += [index]
            elif value < t3:
                q2 += [index]
            else:
                q3 += [index]
        #print("key: ",key, "lens q0,q1,q2,q3: ", len(q0), len(q1), len(q2), len(q3))
            temp_dict_with_quartile_lists["q0"] = q0
            temp_dict_with_quartile_lists["q1"] = q1
            temp_dict_with_quartile_lists["q2"] = q2
            temp_dict_with_quartile_lists["q3"] = q3
            return_dict_with_nested_dict_quartile_lists[key] = temp_dict_with_quartile_lists
        #print(return_dict_with_nested_dict_quartile_lists[key].items())

    return  return_dict_with_nested_dict_quartile_lists

### helper to remove randomnly from a list
def remove_random_elements(indices_list: List[int], 
                           num_elements_to_remove: int
                           ) -> List[int]:
    if num_elements_to_remove > len(indices_list):
        raise ValueError("Number of elements to remove exceeds the length of the list")
    
    indices_to_remove = random.sample(range(len(indices_list)), int(num_elements_to_remove))
    updated_indices_list = [value for index, value in enumerate(indices_list) if index not in indices_to_remove]
    
    return updated_indices_list


### kick out randomnly indices until correct length
def kick_out_random_until_required_length(
        dict_complete_quartiles_lists: Dict[str, Dict[str, List[int]]],
        desired_num_of_indices_in_total: int
        ) -> Dict[str, Dict[str, List[int]]]:
    desired_num_of_indices_per_genre = desired_num_of_indices_in_total/NUM_GENRES
    desired_num_of_indices_per_quartile = desired_num_of_indices_per_genre/4 # bc. 4 quartiles

    return_dict_shortened_quartiles_lists = {}
    for key, quartile_indices_dict in dict_complete_quartiles_lists.items():
        return_dict_shortened_quartiles_lists[key] = {}
        for quartile, indices_list in quartile_indices_dict.items():
            
            listlen_per_quartile = len(indices_list)
            if listlen_per_quartile < desired_num_of_indices_per_quartile:
                raise ValueError("Number of desired indices per quartile greater than there is, actually.")
            ## do some math:
            num_elements_to_remove = int(listlen_per_quartile - desired_num_of_indices_per_quartile)
            print("key:", key, "quartile:", quartile, "number of elements to remove", num_elements_to_remove),

            updated_quartile = remove_random_elements(indices_list, num_elements_to_remove)
            print("key:", key, "quartile:", quartile, "len indices:", len(updated_quartile)),
            return_dict_shortened_quartiles_lists[key][quartile] = updated_quartile
    
    print(return_dict_shortened_quartiles_lists[key].items())
    return return_dict_shortened_quartiles_lists

def kick_out_random_until_required_length_in_mixed_dict(
        dict_complete_quartiles_lists: Dict[str, Dict[str, List[int]]],
        desired_fraction_of_indices_in_total: int
        ) -> Dict[str, Dict[str, List[int]]]:


    return_dict_shortened_quartiles_lists = {}
    for key, quartile_indices_dict in dict_complete_quartiles_lists.items():

        return_dict_shortened_quartiles_lists[key] = {}
        for quartile, indices_list in quartile_indices_dict.items():
            
            listlen_per_quartile = len(indices_list)
            num_elements_to_remove = int(round(listlen_per_quartile*(1-desired_fraction_of_indices_in_total))) # bc 0.1 means 0.9 must be removed

            if len(indices_list)-num_elements_to_remove < 25:
                print("Number of desired indices per quartile is smaller than 100.")
                num_elements_to_remove = len(indices_list) - 25 # so at least 100 stay: 4 quartiles*25

            print("key:", key, "quartile:", quartile, "number of elements to remove", num_elements_to_remove),

            updated_quartile = remove_random_elements(indices_list, num_elements_to_remove)
            print("key:", key, "quartile:", quartile, "len indices:", len(updated_quartile)),
            return_dict_shortened_quartiles_lists[key][quartile] = updated_quartile
    
    print(return_dict_shortened_quartiles_lists.items())
    return return_dict_shortened_quartiles_lists

def save_dicts_updated_indices_per_genre_and_quartile(
        out_file_name: str, 
        data: Dict[str, Dict[str, List[int]]])-> None:
    # File path to save the JSON file
    out_path_obj= os.path.join(
        OS_PATH_STATISTICS_OUTPUT,
        out_file_name
    )
    # Save the dictionary as a JSON file
    with open(out_path_obj, "w") as out_file_obj:
        json.dump(data, out_file_obj, indent=4)

    return None

### main-help function to combine
def get_indices_per_quartiles(
        DESIRED_NUM_OF_INDICES_PER_CLASS_HW,
        DESIRED_NUM_OF_INDICES_PER_CLASS_MG,
        save_indices_dictionary: bool
    ) -> Union[Dict[str, Dict[str, List[int]]],Dict[str, Dict[str, List[int]]]]:
    token_lens_dict_hw = load_json_tokenlengths(FILENAME_TOKENLENGTH_DICT_HW)
    token_lens_dict_mg = load_json_tokenlengths(FILENAME_TOKENLENGTH_DICT_MG_LLAMA)
    dict_complete_quartiles_indices_lists_hw = sort_indices_per_genre_and_quartile(token_lens_dict_hw)
    dict_complete_quartiles_indices_lists_mg = sort_indices_per_genre_and_quartile(token_lens_dict_mg)
    
    if DESIRED_NUM_OF_INDICES_PER_CLASS_HW == None:
        dict_shortened_quartiles_indices_lists_hw = dict_complete_quartiles_indices_lists_hw
    else:
        dict_shortened_quartiles_indices_lists_hw = kick_out_random_until_required_length(
            dict_complete_quartiles_indices_lists_hw,
            DESIRED_NUM_OF_INDICES_PER_CLASS_HW
        ) 
    if save_indices_dictionary:
        save_dicts_updated_indices_per_genre_and_quartile(FILENAME_JSON_INDICES_DICT_HW_TEST_FIX,
                                                        dict_shortened_quartiles_indices_lists_hw
                                                        )
    if DESIRED_NUM_OF_INDICES_PER_CLASS_MG == None:
        dict_shortened_quartiles_indices_lists_mg = dict_complete_quartiles_indices_lists_mg
    else:
        dict_shortened_quartiles_indices_lists_mg = kick_out_random_until_required_length(
            dict_complete_quartiles_indices_lists_mg,
            DESIRED_NUM_OF_INDICES_PER_CLASS_MG
        )
    if save_indices_dictionary:
        save_dicts_updated_indices_per_genre_and_quartile(FILENAME_JSON_INDICES_DICT_MG_TEST_FIX_LLAMA,
                                                        dict_shortened_quartiles_indices_lists_mg
                                                        )
    
    return dict_shortened_quartiles_indices_lists_hw, dict_shortened_quartiles_indices_lists_mg

def get_indices_per_quartiles_additional_models(
        DESIRED_NUM_OF_INDICES_TEST,
        DESIRED_NUM_OF_INDICES_TRAIN,
        save_indices_dictionary: bool
    ) -> Union[Dict[str, Dict[str, List[int]]],Dict[str, Dict[str, List[int]]]]:
    token_lens_dict_test = load_json_tokenlengths(FILENAME_TOKENLENGTH_DICT_MG_TEST_ONLY)
    token_lens_dict_mg = load_json_tokenlengths(FILENAME_TOKENLENGTH_DICT_MG_MIXED)
    dict_complete_quartiles_indices_lists_test = sort_indices_per_genre_and_quartile(token_lens_dict_test)
    dict_complete_quartiles_indices_lists_mg = sort_indices_per_genre_and_quartile(token_lens_dict_mg)
    
    if DESIRED_NUM_OF_INDICES_TEST == None:
        dict_shortened_quartiles_indices_lists_test = dict_complete_quartiles_indices_lists_test
    else:
        dict_shortened_quartiles_indices_lists_test = kick_out_random_until_required_length(
            dict_complete_quartiles_indices_lists_test,
            DESIRED_NUM_OF_INDICES_TEST
        ) 
    if save_indices_dictionary:
        save_dicts_updated_indices_per_genre_and_quartile(FILENAME_JSON_INDICES_DICT_MG_COMPLETE_DATASET_MIXED_TEST_ONLY,
                                                        dict_shortened_quartiles_indices_lists_test
                                                        )
    if DESIRED_NUM_OF_INDICES_TRAIN == None:
        dict_shortened_quartiles_indices_lists_mg = dict_complete_quartiles_indices_lists_mg
    else:
        dict_shortened_quartiles_indices_lists_mg = kick_out_random_until_required_length(
            dict_complete_quartiles_indices_lists_mg,
            DESIRED_NUM_OF_INDICES_TRAIN
        )
    if save_indices_dictionary:
        save_dicts_updated_indices_per_genre_and_quartile(FILENAME_JSON_INDICES_DICT_MG_COMPLETE_DATASET_MIXED,
                                                        dict_shortened_quartiles_indices_lists_mg
                                                        )
    
    return dict_shortened_quartiles_indices_lists_test, dict_shortened_quartiles_indices_lists_mg

def get_indices_per_quartiles_additional_models_test(
        DESIRED_NUM_OF_INDICES_TEST_FRACTION,
        save_indices_dictionary: bool
    ) -> Union[Dict[str, Dict[str, List[int]]],Dict[str, Dict[str, List[int]]]]:
    token_lens_dict_mg = load_json_tokenlengths(FILENAME_TOKENLENGTH_DICT_MG_MIXED)

    dict_complete_quartiles_indices_lists_mg = sort_indices_per_genre_and_quartile(token_lens_dict_mg)


    dict_shortened_quartiles_indices_lists_mg = kick_out_random_until_required_length_in_mixed_dict(
        dict_complete_quartiles_indices_lists_mg,
        DESIRED_NUM_OF_INDICES_TEST_FRACTION
    )
    if save_indices_dictionary:
        save_dicts_updated_indices_per_genre_and_quartile(FILENAME_JSON_INDICES_DICT_MG_TEST_FIX_MIXED,
                                                        dict_shortened_quartiles_indices_lists_mg
                                                        )
    
    return dict_shortened_quartiles_indices_lists_mg

if __name__ == "__main__":
    #get_indices_per_quartiles(2500,2501,True)
    #get_indices_per_quartiles_additional_models(None,None,True)
    get_indices_per_quartiles_additional_models_test(0.1,True)