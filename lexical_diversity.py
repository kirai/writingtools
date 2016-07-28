import nltk
from nltk import pos_tag, word_tokenize
from nltk import FreqDist

# All words average
def lexical_diversity(text):
    return len(text) / len(set(text))

# Percentage of text taken over by one word
def percentage(count, total):
    return 100 * count / total

# Most frequent words
def most_frequent_words(text, num = 3):
    return list(FreqDist(text).keys())[:num]

# Hapaxes
def hapaxes(text, num = 3):
    return list(FreqDist(text).hapaxes())[:num]

# Longest words
def long_words(text, n = 10):
    return [w for w in list(set(text)) if len(w) > n]

def frequent_long_words(text, min_w_length = 4, min_w_ocurrences = 2):
    """Most frequent long words in a given text
       text             -- nltk.Text
       min_w_length     -- minimum word length that will be filtered
       min_w_ocurrences -- minimum number of ocurrences of the long word in the whole text    
    """

    return [w for w in list(set(text)) if len(w) > min_w_length and FreqDist(text)[w] >= min_w_ocurrences]

if __name__ == "__main__":
    text = "The tomato is delicious and the arrow flies like an elephant. I like animals, specially elephants. australopitecusapharensis this is weird tomato I like tomato a lot and animals an elephants"
    tokens = word_tokenize(text)
    text = nltk.Text(tokens)

    print("Analyzing text:")
    print(text)
   
    print("Frequency distribution:")
    print(FreqDist(text).__str__)

    # Most frequent words
    print("Most frequent words are:")
    print(most_frequent_words(text))

    # Hapaxes
    print("Hapaxes are:")
    print(hapaxes(text))

    # Longest words
    print("Longest words are:")
    print(long_words(text))

    # Frequent long words
    print("Most frequent long words are:")
    print(frequent_long_words(text))
