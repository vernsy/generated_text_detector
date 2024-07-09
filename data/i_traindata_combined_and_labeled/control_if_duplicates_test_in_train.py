def check_duplicates(file1, file2):
    duplicates = set()
    #duplicates =0
    # Read lines from file1 and store them in a set
    with open(file2, 'r', encoding='latin-1') as f1:
        lines1 = set(f1.readlines())

    # Read lines from file2 and check if they're duplicates
    with open(file1, 'r',encoding='latin-1') as f2:
        for line in f2:
            if line in lines1:
                duplicates.add(line.strip().decode('utf-8'))
                #duplicates +=1

    return duplicates

# file1 = 'hw_test_texts.jsonl'
# file2 = 'hw_train_eval_texts.jsonl'
file1 = 'mg_test_texts_llama.jsonl'
file2 = 'mg_train_eval_texts_llama.jsonl'

result = check_duplicates(file2, file1)
if result:
    print("Duplicates found:")
    for line in result:
        print(line)
    print("duplicates: ", len(result))
else:
    print("No duplicates found.")