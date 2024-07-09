def count_duplicates(file_path):
    duplicates_count = {}
    
    # Read lines from the file and count duplicates
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if line in duplicates_count:
                duplicates_count[line] += 1
            else:
                duplicates_count[line] = 1

    # Filter duplicates (appearing more than once)
    duplicates = {line: count for line, count in duplicates_count.items() if count > 1}
    
    return duplicates

file_path = 'hw_test_texts.jsonl'
duplicates = count_duplicates(file_path)

if duplicates:
    print("Duplicates found:")
    for line, count in duplicates.items():
        print(f"'{line}' appears {count} times.")
else:
    print("No duplicates found.")