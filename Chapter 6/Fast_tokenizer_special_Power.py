# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))

# %%
tokenizer.is_fast
# %%
encoding.is_fast
# %%
encoding.tokens()
# %%
encoding.word_ids()
# %%
# Try it out! Create a tokenizer from the bert-base-cased and roberta-base checkpoints and tokenize ”81s” with them. What do you observe? What are the word IDs?

from transformers import AutoTokenizer

tokenizer1 = AutoTokenizer.from_pretrained("roberta-base")

# %%
bert_output = tokenizer("81s")
print("bert output", bert_output)
roberta_output = tokenizer1("81s")
print("roberta_output", roberta_output)

# %%
start, end = encoding.word_to_chars(3)
print(example[start:end])
# %%
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)
# %%
print(inputs["input_ids"].shape)
print(outputs.logits.shape)
# %%
import torch

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)
# %%
print(model.config.id2label)
# %%
results = []
tokens = inputs.tokens()
for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
            }
        )
results
# %%
# encoding.word_ids()
inputs.word_to_chars
tokens = inputs.tokens()
len(tokens)

# %%
inputs.tokens()
# %%
