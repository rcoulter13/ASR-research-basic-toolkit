"""
Name: Riah Coulter
Date: 6-22-21
Purpose:
    Create custom dict.ltr.txt which is a dictionary of all letter counts
    for a given dataset prior wav2vec2 training/validation.


Parameters:
-> directory of train.wrd and valid.wrd from modLibri_labels.py output
-> directory of output file to write to

Output:
-> txt file with letter\scount pairs
"""
import argparse
import pandas as pd
import os
import re
import operator

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
    
    letters = {}
    for s in sentence:
      # Replace whitespace with |
      s = re.sub(" ", "|", s)
      for letter in s:
        if letter in letters:
          letters[letter] += 1
        else:
          letters[letter] = 1

    sorted_dict = sorted(letters.items(), key=operator.itemgetter(1), reverse=True)
    assert len(sorted_dict) != 0, "Issue loading in data. Letter dictionary empty!"

    file_to_save = f'{args.output_dir}/dict.ltr.txt'
    file = open(file_to_save, 'w', encoding='utf-8')
    for item in sorted_dict:
      file.write(item[0] + " " + str(item[1]) + "\n")
    file.close()

if __name__ == "__main__":
    main()
