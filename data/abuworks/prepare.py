import os
import numpy as np
import pandas as pd
from maercbpe import RegexTokenizer

# Load your saved tokenizer model
tokenizer = RegexTokenizer()
tokenizer.load("data/abuworks/models/regex.model")

# Load your dataset
with open('data/abuworks/works.txt') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Encode the dataset using your custom tokenizer
train_ids = tokenizer.encode_biasa(train_data)
val_ids = tokenizer.encode_biasa(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


