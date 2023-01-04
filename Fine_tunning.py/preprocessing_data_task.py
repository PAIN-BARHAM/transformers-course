# Build a model for sst2 dataset

# %% 
from datasets import load_dataset

# %% 
raw_datasets = load_dataset("glue", "sst2")
print(raw_datasets)

# %% 
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]


# %% 
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences = tokenizer(raw_datasets["train"]["sentence"])
# tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence"])

inputs = tokenizer("This is the first sentence.")
inputs


# %% 
tokenizer.convert_ids_to_tokens(inputs["input_ids"])

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence"],
    padding=True,
    truncation=True,
)

# %% 
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets

# %% 
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:20]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence"]}
[len(x) for x in samples["input_ids"]]


# %% 

batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}

# %%
