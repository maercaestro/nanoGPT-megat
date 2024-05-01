import os
import requests
import tiktoken
import numpy as np
import json
import re


# Function to remove the first 8 words from a string
def remove_first_8_words(text):
    words = text.split()
    filtered_words = words[8:]
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
print(data)