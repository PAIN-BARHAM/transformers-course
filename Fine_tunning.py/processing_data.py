# %%
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# %%
# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# %%
# This is new
batch["labels"] = torch.tensor([1, 1])

# %%
optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
# %%

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")

raw_datasets
# %%

raw_train_dataset = raw_datasets["train"]

raw_train_dataset[15]
# %%
raw_train_dataset.features
# %%

raw_validation_dataset = raw_datasets["validation"]
raw_validation_dataset[87]
# %%

print(raw_train_dataset[15]["sentence1"])
print(raw_train_dataset[15]["sentence2"])

# %%

print(tokenizer(raw_train_dataset[15]["sentence1"], padding= True, truncation=True, return_tensors='pt'))
print(tokenizer(raw_train_dataset[15]["sentence2"], padding= True, truncation=True, return_tensors='pt'))

# %%
print([tokenizer(raw_train_dataset[15]["sentence1"], padding= True, truncation=True, return_tensors='pt'), tokenizer(raw_train_dataset[15]["sentence2"], padding= True, truncation=True, return_tensors='pt')])

# %%

inputs = tokenizer("This is the first sentence.", "This is the second one.")

tokenizer.convert_ids_to_tokens(inputs["input_ids"])

# %%

tokenized_dataset = tokenizer(

    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding= True,
    truncation= True
)


# %%
tokenized_dataset
# %%

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# %%

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

tokenized_datasets


# %%
from transformers import DataCollatorWithPadding


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


samples = tokenized_datasets["train"][:8]

samples = {k: v for k,v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

[len(x) for x in samples["input_ids"]]

# %%

batch = data_collator(samples)

{k: v.shape for k,v in batch.items()} 

# %%

