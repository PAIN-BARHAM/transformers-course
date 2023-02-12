# %%

from collections import defaultdict

text = [("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)]


alphabets = []

alphabets_dict = defaultdict(int)
all_freq = 0

for word, freq in text:
    # print(word, freq)
    # for alpha in word:
    #     if alpha not in alphabets:
    #         alphabets.append(alpha)

    check_word = word

    print("Length of word is ", len(word))

    for i in range(1, len(word)):

        print(i)
        j = 0
        while j + i <= len(word):
            if word[j : j + i] not in alphabets:
                alphabets.append(word[j : j + i])
            alphabets_dict[word[j : j + i]] += freq
            all_freq += freq
            # print(word[j : j + i])
            j = j + 1
print(alphabets)

print(alphabets_dict)
# %%
alphabets_dict["hug"] += 10
all_freq += 10
# %%

alphabets_dict["ug"]/all_freq
# %%
