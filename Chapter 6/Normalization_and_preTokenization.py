# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))

# %%
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(type(tokenizer.backend_tokenizer))
# %%
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

# %%
