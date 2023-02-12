# %%
from datasets import load_dataset

raw_datasets = load_dataset("conll2003")

# %%
raw_datasets

# %%
raw_datasets["train"][0]["tokens"]

# %%

raw_datasets["train"][0]["ner_tags"]

# %%

ner_feature = raw_datasets["train"].features["ner_tags"]

ner_feature
# %%

label_names = ner_feature.feature.names
label_names
# %%
