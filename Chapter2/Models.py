# %%
from transformers import BertConfig, BertModel

# %%
# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)


# %%

print(config)

# %%
# Import a pretrained model

from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")


# %%
model.save_pretrained("./models_files")

# %%

encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

# %%

import torch
model_inputs = torch.tensor(encoded_sequences)


# %%

tokenized_text = "Jim's Henson! was a puppeteer".split()
print(tokenized_text)


# %%
