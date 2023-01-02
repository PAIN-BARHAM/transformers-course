# %% 
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import copy


# %%
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# %% 
sequence = "I've been waiting for a HuggingFace course my whole life."


# %%
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([ids])

# This line will fail.
model(input_ids)
# %%

tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
# %%

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)
# %%
# Test 

batched_ids = [ids, ids]
input_ids = torch.tensor(batched_ids)

print(input_ids)

output = model(input_ids)

print(output.logits)
# %%
## Padding 

padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]

# %%
print(tokenizer.pad_token_id)
# %%

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],[200, 200, tokenizer.pad_token_id],]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
# %%
batched_ids = [
    [200,200,200],
    [200,200,tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1,1,0],
]

outputs = model( torch.tensor(batched_ids), attention_mask= torch.tensor(attention_mask))

print(outputs)

# %%
# Test 

sentence1 = "I’ve been waiting for a HuggingFace course my whole life."
sentence2 = "I hate this so much!"

tokens1 = tokenizer.tokenize(sentence1)
print(tokens1)

ids1 = tokenizer.convert_tokens_to_ids(tokens1)
print(ids1)


tokens2 = tokenizer.tokenize(sentence2)
print(tokens2)

ids2 = tokenizer.convert_tokens_to_ids(tokens2)
print(ids2)

# %% 
# batch_sentences= [sentence1, sentence2]

# tockens_total = tokenizer.tokenize(batch_sentences)
# ids_total = tokenizer.convert_tokens_to_ids(tockens_total)
ids_total = [ids1, ids2]
tockens_total = [tokens1, tokens2]

print(tockens_total)
print(ids_total)
# %%
# Predict the output of the test

print(model(torch.tensor([ids1])))
print(model(torch.tensor([ids2])))

print(len(ids1))
print(len(ids2))



# %% 

ids_total_before = copy.deepcopy(ids_total)
print("Before", ids_total_before)

# %% 

max_len = 0 

for item in ids_total:
    if len(item) > max_len:
        max_len = len(item)

for item in ids_total:
    if len(item) < max_len:
        listofzeros = [0] * (max_len - len(item))
        item.extend(listofzeros)
        print(item)

ids_total
# %% 


attention_mask = [] 


# %% 

# Predicts for a long sequesnce:

for item in ids_total_before:
    print(item)
    if len(item) == max_len:
        attention_mask.append([1]*len(item))
        print(attention_mask)
    if len(item) < max_len:
        ls = [1] * len(item)
        diff= max_len-len(item)
        ls2 = [0]*diff
        ls.extend(ls2)
        attention_mask.append(ls)


# %% 

model(torch.tensor(ids_total), attention_mask = torch.tensor(attention_mask))


# %%
