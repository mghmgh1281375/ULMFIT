
# Preprocess data using Hazm

import pandas as pd
import re, os
from termcolor import colored, cprint
from hazm import WordTokenizer, Normalizer
tokenizer = WordTokenizer(separate_emoji=False, replace_links=False, replace_IDs=False, replace_emails=False, replace_numbers=False, replace_hashtags=False)
normalizer = Normalizer()

def remove_section(filename):
    pattern = re.compile(r'Section\:+')
    with open(filename, 'r') as file:
        with open(os.path.join(os.path.dirname(filename), f'without_section.{os.path.basename(filename)}'), 'w') as out:
            line = file.readline()
            while line:
                out.write(pattern.sub(string=line, repl=''))
                line = file.readline()

    return os.path.join(os.path.dirname(filename), f'without_section.{os.path.basename(filename)}')

if __name__ == "__main__":

    # Remove Section:::: 
    # filname = remove_section('resources/fa.wiki.small.csv')
    filname = remove_section('resources/fa.wiki.csv')
    # df = pd.read_csv(filname)#, nrows=20)

    # # Normalize
    # df = pd.read_csv('resources/fa.wiki.csv')#, nrows=20)
    # print(df.columns, df.head(1), len(df))
    # s = df.apply(lambda x: normalizer.normalize(x.tolist()[0]), axis=1)
    # print(s.head(1), len(s))
    # s.to_csv('resources/fa.wiki.normalized.csv', header=['text'], index=False)

    # # Tokenize
    # df = pd.read_csv('resources/fa.wiki.normalized.csv')#, nrows=20)
    # print(df.columns, df.head(1), len(df))
    # s = df.apply(lambda x: ' '.join(tokenizer.tokenize(x.tolist()[0])), axis=1)
    # print(s.head(1), len(s))
    # s.to_csv('resources/fa.wiki.normalized.tokenized.csv', header=['text'], index=False)

    print(colored('Successful!', 'green', attrs=['reverse', 'blink']))
