#%%

from transformers import AutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)


#%%

distilbert_num_parameters= model.num_parameters() /1_000_000
print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")
# %%

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# %%
text = "This is a great [MASK]."

inputs = tokenizer(text, return_tensors ="pt")
inputs
# %%
import torch

token_logits = model(**inputs).logits
token_logits

# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

mask_token_index

# %%
mask_token_logits = token_logits[0, mask_token_index, :]
mask_token_logits
# %%
# Pick the [MASK] candidates with the highest logits

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

top_5_tokens
# %%
for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

# %%
## Use IMDb

from datasets import load_dataset 


imdb_dataset = load_dataset("imdb")
imdb_dataset 

# %%

sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")
# %%
sample = imdb_dataset["unsupervised"].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> Review: {row['text']}'")
    print(f"'>>> Label: {row['label']}'")
# %%
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result
# %%
# Use batched=True to activate fast multithreading!
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
tokenized_datasets
# %%
tokenizer.model_max_length
# %%
from transformers import AutoTokenizer

bigbird_tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
# %%
bigbird_tokenizer.model_max_length
# %%
chunk_size = 128

# Slicing produces a list of lists for each feature
tokenized_samples = tokenized_datasets["train"][:3]
# %%
for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f"'>>> Review {idx} length: {len(sample)}'")
# %%
concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
# %%
concatenated_examples
# %%
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")

# %%
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}

for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")
# %%
