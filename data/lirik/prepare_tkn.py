import os
import requests
import tiktoken
import numpy as np
import json
import re


import json
import re

# Function to remove the first 8 words from a string
def remove_first_8_words(text):
    words = text.split()
    filtered_words = words[8:]
    
    # Check if '\n' exists and add punctuation before it
    for i, word in enumerate(filtered_words):
        if '\n' in word:
            # Split the word at '\n'
            parts = word.split('\n')
            # Add punctuation to the first part of the word
            parts[0] += '.'
            # Join the parts back together
            filtered_words[i] = '\n'.join(parts)
    
    return ' '.join(filtered_words)

# Load JSONL data from file and process each line
modified_texts = []
with open('data/lirik/lirik.jsonl', 'r') as file:
    for line in file:
        json_data = json.loads(line)
        text = json_data["body"]  # Adjust this according to your JSONL structure
        modified_text = remove_first_8_words(text)
        modified_texts.append(modified_text)

# Join all the modified strings into a single string
data = ' '.join(modified_texts)


n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
