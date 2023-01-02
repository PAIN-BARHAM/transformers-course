# %%
from transformers import AutoTokenizer
from pprint import pprint

# %%
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# %%
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
# %%
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

# print(inputs)
# print(tokenizer.vocab["'"])

for input_ids in inputs['input_ids']:
    for item in input_ids:
    # print(int(item))
        print(tokenizer.decode(int(item)))

# for one item 
for item in inputs['input_ids'][0]:
    print(tokenizer.decode(int(item)))


# pprint(inputs)

# %%
from transformers import AutoModel

# %%
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
# %%
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
# %%

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
# %%
import torch 

predictions = torch.nn.functional.softmax(outputs.logits, dim =-1)

print(predictions)

print(model.config.id2label)

# %%
