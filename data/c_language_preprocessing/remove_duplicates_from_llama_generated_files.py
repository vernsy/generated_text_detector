def remove_duplicates(input_file, output_file):
    unique_lines = set()

    # Read lines from the input file and store unique lines
    with open(input_file, 'r') as f_input:
        for line in f_input:
            unique_lines.add(line.strip())

    # Write unique lines to the output file
    with open(output_file, 'w') as f_output:
        for line in unique_lines:
            f_output.write(line + '\n')

input_file = '/data/verena/llama_generated_from_zeitonline_new.jsonl'
output_file = '/data/verena/llama_generated_from_zeitonline_new_no_duplicates.jsonl'
remove_duplicates(input_file, output_file)