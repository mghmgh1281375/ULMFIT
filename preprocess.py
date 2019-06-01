
# Preprocess data using Hazm

import pandas as pd
from hazm import WordTokenizer, Normalizer
tokenizer = WordTokenizer(separate_emoji=False, replace_links=False, replace_IDs=False, replace_emails=False, replace_numbers=False, replace_hashtags=False)
normalizer = Normalizer()



if __name__ == "__main__":
    df = pd.read_csv('resources/fa.wiki.csv')
    # print(df.columns)
    print(df.head(), len(df))
    df = df.apply(lambda x: normalizer.normalize(x.tolist()[0]), axis=1)
    print(df.head(), len(df))
    df.to_csv('resources/fa.wiki.normalized.csv')

