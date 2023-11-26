import json

def convert_to_jsonl(input_file_path, output_file_path):
    # Read the file and process each line
    with open(input_file_path, 'r') as file:
        # Initialize an empty list to store messages
        messages = []

        for line in file:
            # Remove leading and trailing whitespaces from each line
            cleaned_line = line.strip()

            # Create a message dictionary
            message = {"role": "assistant", "content": cleaned_line}

            # Create the final JSON structure
            json_data = {"messages": [{"role": "user", "content": "You are a British football commentator. Generate the commentary of an exciting game."}, message]}

            # Append the JSON data to the list
            messages.append(json_data)

    # Write the JSON data to the output file in JSON Lines format
    with open(output_file_path, 'w') as output_file:
        for message in messages:
            json_line = json.dumps(message)
            output_file.write(json_line + '\n')

# Specify the paths to your input and output files
input_file_path = 'transcripts.txt'
output_file_path = 'commentary_train_whisper.jsonl'

# Call the function to convert and write to the output file
# convert_to_jsonl(input_file_path, output_file_path)


import re
import jsonlines

def create_jsonl(transcription_sub):
    prompt = "You are a British football commentator. Generate the commentary of an exciting game."
    message = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": transcription_sub}
    ]
    return {"messages": message}

# Input file with transcribed lines
input_file_path = "transcripts.txt"  # Replace with the path to your file

# Output JSONL file
output_file_path = "commentary_train_whisper_2_sentences.jsonl"

# Function to split lines into two-sentence chunks
def split_into_chunks(line):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', line)
    return [sentences[i:i+2] for i in range(0, len(sentences), 2)]

with open(input_file_path, 'r') as input_file, jsonlines.open(output_file_path, mode='w') as writer:
    for line in input_file:
        chunks = split_into_chunks(line)
        for chunk in chunks:
            transcription_sub = '. '.join(chunk)
            json_entry = create_jsonl(transcription_sub)
            writer.write(json_entry)

