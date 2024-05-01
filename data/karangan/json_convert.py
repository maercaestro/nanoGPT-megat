import json
import os

# Python program to read
# json file
 
with open('karangan.net.json', 'r') as json_file:
    data = json.load(json_file)
data = ''.join(''.join(sub_entry) for entry in data for sub_entry in entry["p"])

data