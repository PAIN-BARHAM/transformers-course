# %%

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# %%

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# %%


from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text
    )
    # print(words_with_offsets)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)

# %%

alphabet = []

for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)

# print("Before sorting", alphabet)

alphabet.sort()

print("After sorting", alphabet)

# %%

vocab = ["<|endoftext|>"] + alphabet.copy()
vocab
# %%
# {word: {'a', 'b', 'c'}}


splits = {word: [c for c in word] for word in word_freqs.keys()}
splits
# %%

# input => This is the
# output = (T, h), (h, i), (i, s), (G, i), (i, s), (Ġ, t), (t, h), (h, e)



def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)

    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq

    return pair_freqs


# %%

pair_freqs = compute_pair_freqs(splits)

for i, key in enumerate(pair_freqs):
    print(f"{key}: {pair_freqs[key]}")


# %%

max_freq = 0
best_pair = ""

for pair, freq in pair_freqs.items():

    if freq > max_freq:
        best_pair = pair
        max_freq = freq

best_pair, max_freq

# %%
merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")

# %%


def merge_pair(a, b, splits):

    for word in word_freqs:

        # characters in the splits of word
        split = splits[word]

        # Edge case, for example space at the end of the scentecte
        if len(split) == 1:
            continue

        # print("New Split")
        i = 0
        while i < len(split) - 1:
            # print("Best pairs", split)
            # print("len(split)", len(split))
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
                splits[word] = split
            else:
                i += 1
    return splits


# %%
splits = merge_pair("Ġ", "t", splits)

print(splits["Ġtrained"])

# %%

vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)

    max_freq = None
    best_pair = ""

    for pair, freq in pair_freqs.items():

        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    print("Best pair", best_pair)

    splits = merge_pair(*best_pair, splits)

    print("Splits ")
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])
# %%
print("Merges")
merges


# %%
print("Merges")
vocab
# %%

# %%

# Input => "This is not a token."
# Output => ['This', 'Ġis', 'Ġ', 'n', 'o', 't', 'Ġa', 'Ġtoken', '.']


def tokenize(Text):

    # Get the words and offsets
    pre_tokenize_result = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        Text
    )

    # Get the words from the tokenized resluts
    pre_tokenized_text = [word for word, _ in pre_tokenize_result]

    # Get the splitted words as characters classified in lists
    splits = [[l for l in word] for word in pre_tokenized_text]

    for pair, merge in merges.items():
        for key_word, split in enumerate(splits):

            i = 0

            while i < len(split) - 1:

                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[key_word] = split

    return splits

# %%
tokenize("This is not a token.")

# %%


for txt in corpus:
    print(tokenize(txt))
# %%



# %%


# word_freqs_1

{'This': 3, 'Ġis': 2, 'Ġthe': 1, 'ĠHugging': 1, 'ĠFace': 1, 'ĠCourse': 1, '.': 4, 'Ġchapter': 1, 'Ġabout': 1, 'Ġtokenization': 1, 'Ġsection': 1, 'Ġshows': 1, 'Ġseveral': 1, 'Ġtokenizer': 1, 'Ġalgorithms': 1, 'Hopefully': 1, ',': 1, 'Ġyou': 1, 'Ġwill': 1, 'Ġbe': 1, 'Ġable': 1, 'Ġto': 1, 'Ġunderstand': 1, 'Ġhow': 1, 'Ġthey': 1, 'Ġare': 1, 'Ġtrained': 1, 'Ġand': 1, 'Ġgenerate': 1, 'Ġtokens': 1}


# alphabet_1 
[',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'Ġ']


# %%
# Splits at the beginnning
# Splits
{'This': ['T', 'h', 'i', 's'],
 'Ġis': ['Ġ', 'i', 's'],
 'Ġthe': ['Ġ', 't', 'h', 'e'],
 'ĠHugging': ['Ġ', 'H', 'u', 'g', 'g', 'i', 'n', 'g'],
 'ĠFace': ['Ġ', 'F', 'a', 'c', 'e'],
 'ĠCourse': ['Ġ', 'C', 'o', 'u', 'r', 's', 'e'],
 '.': ['.'],
 'Ġchapter': ['Ġ', 'c', 'h', 'a', 'p', 't', 'e', 'r'],
 'Ġabout': ['Ġ', 'a', 'b', 'o', 'u', 't'],
 'Ġtokenization': ['Ġ',
  't',
  'o',
  'k',
  'e',
  'n',
  'i',
  'z',
  'a',
  't',
  'i',
  'o',
  'n'],
 'Ġsection': ['Ġ', 's', 'e', 'c', 't', 'i', 'o', 'n'],
 'Ġshows': ['Ġ', 's', 'h', 'o', 'w', 's'],
 'Ġseveral': ['Ġ', 's', 'e', 'v', 'e', 'r', 'a', 'l'],
 'Ġtokenizer': ['Ġ', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'e', 'r'],
 'Ġalgorithms': ['Ġ', 'a', 'l', 'g', 'o', 'r', 'i', 't', 'h', 'm', 's'],
 'Hopefully': ['H', 'o', 'p', 'e', 'f', 'u', 'l', 'l', 'y'],
 ',': [','],
 'Ġyou': ['Ġ', 'y', 'o', 'u'],
 'Ġwill': ['Ġ', 'w', 'i', 'l', 'l'],
 'Ġbe': ['Ġ', 'b', 'e'],
 'Ġable': ['Ġ', 'a', 'b', 'l', 'e'],
 'Ġto': ['Ġ', 't', 'o'],
 'Ġunderstand': ['Ġ', 'u', 'n', 'd', 'e', 'r', 's', 't', 'a', 'n', 'd'],
 'Ġhow': ['Ġ', 'h', 'o', 'w'],
 'Ġthey': ['Ġ', 't', 'h', 'e', 'y'],
 'Ġare': ['Ġ', 'a', 'r', 'e'],
 'Ġtrained': ['Ġ', 't', 'r', 'a', 'i', 'n', 'e', 'd'],
 'Ġand': ['Ġ', 'a', 'n', 'd'],
 'Ġgenerate': ['Ġ', 'g', 'e', 'n', 'e', 'r', 'a', 't', 'e'],
 'Ġtokens': ['Ġ', 't', 'o', 'k', 'e', 'n', 's']}


pair_freqs = 
('T', 'h'): 3
('h', 'i'): 3
('i', 's'): 5
('Ġ', 'i'): 2
('Ġ', 't'): 7
('t', 'h'): 3
('h', 'e'): 2
('Ġ', 'H'): 1
('H', 'u'): 1
('u', 'g'): 1
('g', 'g'): 1
('g', 'i'): 1
('i', 'n'): 2
('n', 'g'): 1
('Ġ', 'F'): 1
('F', 'a'): 1
('a', 'c'): 1
('c', 'e'): 1
('Ġ', 'C'): 1
('C', 'o'): 1
('o', 'u'): 3
('u', 'r'): 1
('r', 's'): 2
('s', 'e'): 3
('Ġ', 'c'): 1
('c', 'h'): 1
('h', 'a'): 1
('a', 'p'): 1
('p', 't'): 1
('t', 'e'): 2
('e', 'r'): 5
('Ġ', 'a'): 5
('a', 'b'): 2
('b', 'o'): 1
('u', 't'): 1
('t', 'o'): 4
('o', 'k'): 3
('k', 'e'): 3
('e', 'n'): 4
('n', 'i'): 2
('i', 'z'): 2
('z', 'a'): 1
('a', 't'): 2
('t', 'i'): 2
('i', 'o'): 2
('o', 'n'): 2
('Ġ', 's'): 3
('e', 'c'): 1
('c', 't'): 1
('s', 'h'): 1
('h', 'o'): 2
('o', 'w'): 2
('w', 's'): 1
('e', 'v'): 1
('v', 'e'): 1
('r', 'a'): 3
('a', 'l'): 2
('z', 'e'): 1
('l', 'g'): 1
('g', 'o'): 1
('o', 'r'): 1
('r', 'i'): 1
('i', 't'): 1
('h', 'm'): 1
('m', 's'): 1
('H', 'o'): 1
('o', 'p'): 1
('p', 'e'): 1
('e', 'f'): 1
('f', 'u'): 1
('u', 'l'): 1
('l', 'l'): 2
('l', 'y'): 1
('Ġ', 'y'): 1
('y', 'o'): 1
('Ġ', 'w'): 1
('w', 'i'): 1
('i', 'l'): 1
('Ġ', 'b'): 1
('b', 'e'): 1
('b', 'l'): 1
('l', 'e'): 1
('Ġ', 'u'): 1
('u', 'n'): 1
('n', 'd'): 3
('d', 'e'): 1
('s', 't'): 1
('t', 'a'): 1
('a', 'n'): 2
('Ġ', 'h'): 1
('e', 'y'): 1
('a', 'r'): 1
('r', 'e'): 1
('t', 'r'): 1
('a', 'i'): 1
('n', 'e'): 2
('e', 'd'): 1
('Ġ', 'g'): 1
('g', 'e'): 1
('n', 's'): 1


# vocab_size will determine the number of merges we have 

# Merges

{('Ġ', 't'): 'Ġt',
 ('i', 's'): 'is',
 ('e', 'r'): 'er',
 ('Ġ', 'a'): 'Ġa',
 ('Ġt', 'o'): 'Ġto',
 ('e', 'n'): 'en',
 ('T', 'h'): 'Th',
 ('Th', 'is'): 'This',
 ('o', 'u'): 'ou',
 ('s', 'e'): 'se',
 ('Ġto', 'k'): 'Ġtok',
 ('Ġtok', 'en'): 'Ġtoken',
 ('n', 'd'): 'nd',
 ('Ġ', 'is'): 'Ġis',
 ('Ġt', 'h'): 'Ġth',
 ('Ġth', 'e'): 'Ġthe',
 ('i', 'n'): 'in',
 ('Ġa', 'b'): 'Ġab',
 ('Ġtoken', 'i'): 'Ġtokeni'}

# Text:

[['This'], ['Ġis'], ['Ġthe'], ['Ġ', 'H', 'u', 'g', 'g', 'in', 'g'], ['Ġ', 'F', 'a', 'c', 'e'], ['Ġ', 'C', 'ou', 'r', 'se'], ['.']]
[['This'], ['Ġ', 'c', 'h', 'a', 'p', 't', 'er'], ['Ġis'], ['Ġab', 'ou', 't'], ['Ġtokeni', 'z', 'a', 't', 'i', 'o', 'n'], ['.']]
[['This'], ['Ġ', 'se', 'c', 't', 'i', 'o', 'n'], ['Ġ', 's', 'h', 'o', 'w', 's'], ['Ġ', 'se', 'v', 'er', 'a', 'l'], ['Ġtokeni', 'z', 'er'], ['Ġa', 'l', 'g', 'o', 'r', 'i', 't', 'h', 'm', 's'], ['.']]
[['H', 'o', 'p', 'e', 'f', 'u', 'l', 'l', 'y'], [','], ['Ġ', 'y', 'ou'], ['Ġ', 'w', 'i', 'l', 'l'], ['Ġ', 'b', 'e'], ['Ġab', 'l', 'e'], ['Ġto'], ['Ġ', 'u', 'nd', 'er', 's', 't', 'a', 'nd'], ['Ġ', 'h', 'o', 'w'], ['Ġthe', 'y'], ['Ġa', 'r', 'e'], ['Ġt', 'r', 'a', 'in', 'e', 'd'], ['Ġa', 'nd'], ['Ġ', 'g', 'en', 'er', 'a', 't', 'e'], ['Ġtoken', 's'], ['.']]

