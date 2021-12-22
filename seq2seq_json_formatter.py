"""
Author: Riah Coulter
Date: 7-13-21
Purpose: Reads in csv from wav2vec prediction output and reformats the target
         and reference predictions into the following json structure:
        {
         translation:
         [{"befr": "alpha bada seven", "en": "alpha beta seven"}
         {"befr": "ten fur okya", "en": "ten four okay"}]
        }
         So that a transformer model (such as mBART or Bert2Bert) that was trained
         on a translation task can be fine-tuned as a Grammar Error Checker (GEC)
         for the target data.
"""
import csv
import json
import argparse
import random
import os

def create_split(contents: list, percentage: float):
    """
    Creates a split in the data based on the percentage
    given. Useful for establishing train/validation/test splits.

    :params: contents - [list] all sentences, percentage - [float] 
    amount to be split off (0.12 == 12%)
    :returns: [list] remaining sentences, [list] sentences split off
    """
    perc_of_items = int(len(contents)*percentage)
    split_indicies = random.sample(range(len(contents)),perc_of_items)
    
    split     = []
    non_split = []
    for i in range(len(contents)):
        if i in split_indicies:
            split.append(contents[i])
        else:
            non_split.append(contents[i])
    return non_split, split
    
def write_to_json(output: dict, output_dir: str, data_type: str, unique_name: str):
    """
    Takes formatted dictionary and writes it to a JSON file.

    :params: [dict] output - dictionary of information, [str] output_dir - directory
    to where the JSON file will be written, [str] data_type - either train, validation
    or test, [str] unique_name - name for new JSON file to be written
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(e)
            raise
    file_path = os.path.join(output_dir, (unique_name + '_seq2seq_' + data_type + '.json'))
    
    with open(file_path, 'w') as file:
        json.dump(output, file)
    print(f"{data_type} json file successfully written!")

def write_to_csv(contents: list, output_dir: str, data_type: str):
    """
    CSV Function depracated.
    """
    name = "seq2seq_" + data_type + ".csv"
    output_file = open(os.path.join(output_dir, name), 'w', encoding='utf-8', newline='')
    writer = csv.writer(output_file)
    #write header
    writer.writerow(['translation'])
    for line in contents:
        writer.writerow(line)
    output_file.close()
    print(f"{data_type} csv file successfully written!")

def main():
    """
    Reads in information from user, reformats data into a large dictionary
    of training pairs, validation pairs and (optional) test pairs to be 
    written to JSON files.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",        required=True,   help="Name to uniquely identify output JSON file")
    parser.add_argument("--files",       required=True,   help="Comma separated string of csv files (or paths to files) to process and reformat")
    parser.add_argument("--output-dir",  required=True,   help="Directory to save new json files to.")
    parser.add_argument("--val-split",   required=True,   help="Percentage (as decimal) of data to be used for validation")
    parser.add_argument("--csvIndices", required=True,   help="Column indices to grab for source-target sentences (ex: 5,6 with 5 being source and 6 target)")
    parser.add_argument("--test-split",                   help="Percentage (as decimal) of data to be used for testing")
    args = parser.parse_args()

    
    all_contents = []
    files = args.files.split(',')
    indices = args.csvIndices.split(',')
    if len(indices) != 2:
        print("Either source or target index missing. Must be separated by a comma: 5,6 with 5 being source and 6 being target.")
        exit()
    sourceIndex = int(indices[0])
    targetIndex = int(indices[1])
    for file in files:
        print(f"Reading {file}...")
        with open(file, mode = 'r') as opened:
            csvFile = csv.reader(opened)
            # format and add to all_contents
            count = 0
            for line in csvFile:
                if count > 0 and line[sourceIndex] != "" and line[targetIndex] != "":
                    dictionary = {"befr": line[sourceIndex], "en": line[targetIndex]}
                    all_contents.append(dictionary)
                else:
                    count += 1
    training_data = all_contents
    if args.val_split is not None:
        training_data, val_data  = create_split(training_data, float(args.val_split))
        validate = {}
        validate["translation"] = val_data
        dataset = {}
        dataset['data'] = validate
        write_to_json(dataset, args.output_dir, "validation", args.name)
    if args.test_split is not None:
        training_data, test_data = create_split(training_data, float(args.test_split))
        test = {}
        test["translation"] = test_data
        write_to_json(test, args.output_dir, "test", args.name)
    train = {}
    train["translation"] = training_data
    dataset = {}
    dataset['data'] = train
    write_to_json(dataset, args.output_dir, "train", args.name)

        
if __name__ == "__main__":
    main()
