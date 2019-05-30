
# Preprocess data using Hazm

from hazm import WordTokenizer, Normalizer
tokenizer = WordTokenizer(separate_emoji=False, replace_links=False, replace_IDs=False, replace_emails=False, replace_numbers=False, replace_hashtags=False)
normalizer = Normalizer()

