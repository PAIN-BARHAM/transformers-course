# %%

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Target: calculate the frequency of each word on the corpus

from collections import defaultdict

word_freqs = defaultdict(int)

for text in corpus:

    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text
    )

    new_words = [word for word, _ in words_with_offsets]

    for word in new_words:
        word_freqs[word] += 1


word_freqs

# %%

# Target: Get the alphabets as irst letters of words, and all the other letters that appear in words prefixed by ##

alphabet = []

for word in word_freqs.keys():

    if word[0] not in alphabet:
        alphabet.append(word[0])

    for letter in word[1:]:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")

alphabet.sort()
alphabet

# %%

vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

# %%
# Splittting the words to the form used by BERT
# First letter without ## and other letters with added ## at the begining

splits = {
    word: [letter if i == 0 else "##" + letter for i, letter in enumerate(word)]
    for word in word_freqs.keys()
}
# print the splits
splits

# %%
# Target --> calculate the score of the pairs
# score=(freq_of_pair)/(freq_of_first_element×freq_of_second_element)
# input is the splits and the output is the scores


def compute_pair_scores(splits):

    # {"letter": freqs}
    letter_freqs = defaultdict(int)
    # {("a", "b"): freqs}
    pair_freqs = defaultdict(int)

    for word, word_list in splits.items():

        if len(word_list) == 1:
            letter_freqs[word] += word_freqs[word]
            continue

        for letter in word_list:
            letter_freqs[letter] += word_freqs[word]

        for i in range(len(word_list) - 1):
            pair_freqs[(word_list[i], word_list[i + 1])] += word_freqs[word]

    scores = {
        pairs: freqs / (letter_freqs[pairs[0]] * letter_freqs[pairs[1]])
        for pairs, freqs in pair_freqs.items()
    }

    return scores


# %%
pair_scores = compute_pair_scores(splits)
for i, key in enumerate(pair_scores.keys()):
    print(f"{key}: {pair_scores[key]}")
    if i >= 5:
        break
# %%

best_pair = ""
max_score = 0

for pair, score in pair_scores.items():

    if score > max_score:
        max_score = score
        best_pair = pair


print(best_pair, max_score)

# %%
vocab.append("ab")
# %%
# Merge the pairs on the splits
from tqdm import tqdm


def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


# %%
splits = merge_pair("a", "##b", splits)

# %%
splits["about"]
# %%

# Do the training to learn from the vocab

vocab_size = 70

while len(vocab) < vocab_size:

    scores = compute_pair_scores(splits)

    best_pair, max_score = "", None

    for pair, score in scores.items():

        if max_score is None or max_score < score:
            max_score = score
            best_pair = pair

    splits = merge_pair(*best_pair, splits)

    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )

    vocab.append(new_token)


print(vocab)

# %%


def encode_word(word):

    tokens = []

    while len(word) > 0:
        i = len(word)

        while i > 0 and word[:i] not in vocab:
            i -= 1

        if i == 0:
            return ["[UNK]"]

        tokens.append(word[:i])

        word = word[i:]

        if len(word) > 0:
            word = f"##{word}"

    return tokens


# %%
print(encode_word("Hugging"))
print(encode_word("HOgging"))

# %%


def tokenize(text):

    pre_tokenize_result = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text
    )
    
    print(pre_tokenize_result)

    pre_tokenized_text = [word for word, offset in pre_tokenize_result]

    print(pre_tokenized_text)

    encode_words = [encode_word(word) for word in pre_tokenized_text]

    return encode_words


# %%
tokenize("This is the Hugging Face course!")


# %%
