"""
Based on code given by Wahyubram82 on bleepcoder

Creates a lexicon for wav2vec2 training/validation data

Note: Slightly modified for efficiency and ease of usage in Colab

"""
import os
import codecs
import re
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--valid_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df1 = pd.read_csv(args.train_dir, header=None)
    df2 = pd.read_csv(args.valid_dir, header=None)

    df1.columns = ['raw']
    df2.columns = ['raw']

    df1 = df1.drop_duplicates('raw',keep='last')
    df2 = df2.drop_duplicates('raw',keep='last')

    sentence1 = df1['raw'].to_list()
    sentence2 = df2['raw'].to_list()
    sentence = sentence1 + sentence2

    word = []
    for x in sentence:
        tmp = x.split(' ')
        for y in tmp:
            if y not in word:
                word.append(y)

    lexicon = []
    for x in range(len(word)):
        wrd = word[x]
        temp = []
        for y in wrd:
            temp.append(y)
        result = ' '.join(temp) + ' |'
        lexicon.append(wrd + '\t ' + result)
    print(f"lexicon: {lexicon}")
    file_to_save = f'{args.output_dir}/lexicon.txt'
    f=codecs.open(file_to_save,'a+','utf8')
    for x in lexicon:
        f.write(x+'\n')
    f.close()

if __name__ == "__main__":
    main()
