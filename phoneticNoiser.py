"""
Author: Riah Coulter
Date: 8-2-21
Purpose: Take in ASR-generated transcripts and noise them phonetically
         (mimicking the consequences of assimilation, insertion/deletion,
         similary manner/place of articulation misidentification, etc.) so 
         that a Grammar Error Checker (GEC) translation task can be conducted 
         with the noised data for training.

         Parameters:
            > --path : Path to txt file containing transcripts
            > --percent : Percent of data to be noised
            > --outpath : Path to which to write noised/unnoised sentence pair csv
            > --output-name : Name for new noised csv file

Types of phonetic noising done by this script:
    -> Assimilation (phonetically combining one word with neighbor)
    -> Similar manner of articulation swap ('v' vs 'f')
    -> Homophone swap ('sent' vs 'scent')


Ideas for further development:
    -> Give user control over noise-type weights
"""
from wordhoard import Homophones
from tqdm import tqdm
import argparse
import random
import time
import nltk
import csv
import re
import os

# If set to true then more spelling errors per word will occur
INTENSE_SPELLING = True

# Variety of common spellings (English), whitespace insertions, manner of articulation
MANNER_MAPPINGS = {
	'a' : ['i','e','u','o'," "],
	'b' : ['p','t','d','k','g'," "],
	'c' : ['p','b','t','d','k','g','f','v','s','z','j','h'," "],
	'd' : ['p','b','t','k','g'," "],
	'e' : ['i','a'," "],
	'f' : ['v','s','z','j','h'," "],
	'g' : ['p','b','t','d','k'," "],
	'h' : ['f','v','s','z','j'," "],
	'i' : ['e','a'," "],
	'j' : ['y','l','r','w','f','v','s','z','j','h'," "],
	'k' : ['p','b','t','d','g'," "],
	'l' : ['j','y','r','w'," "],
	'm' : ['n'," "],
	'n' : ['m'," "],
	'o' : ['u','a'," "],
	'p' : ['b','t','d','k','g'," "],
	'q' : ['p','b','t','d','k','g'," "],
	'r' : ['j','y','l','w'," "],
	's' : ['f','v','z','j','h'," "],
	't' : ['p','b','d','k','g'," "],
	'u' : ['o','a'," "],
	'v' : ['f','s','z','j','h'," "],
	'w' : ['j','y','l','r'," "],
	'x' : ['p','b','t','d','k','g','f','v','s','z','j','h'," "],
	'y' : ['j','l','r','w'," "],
	'z' : ['f','v','s','j','h'," "],
	'bb': ['p','b','t','d','k','g'," "],
    'ch': ['f','v','s','z','j','h'," "],
	'dd': ['p','b','t','d','k','g'," "],
	'ff': ['f','v','s','z','j','h'," "],
	'ph': ['f','v','s','z','j','h'," "],
	'gh': ['f','v','s','z','j','h'," "],
	'lf': ['f','v','s','z','j','h'," "],
	'ft': ['f','v','s','z','j','h'," "],
	'wh': ['f','v','s','z','j','h'," "],
	'dg': ['f','v','s','z','j','h'," "],
	'gg': ['f','v','s','z','j','h'," "],
	'cc': ['f','v','s','z','j','h','p','b','t','d','k','g'," "],
    'ht': ['p','b','t','d','k','g'," "],
	'll': ['j','y','l','r','w'," "],
	'mm': ['m','n'," "],
	'mb': ['m','n'," "],
	'lm': ['m','n'," "],
	'nn': ['m','n'," "],
	'kn': ['m','n'," "],
	'gn': ['m','n'," "],
	'pn': ['m','n'," "],
	'pp': ['p','b','t','d','k','g'," "],
	'ph': ['f','v','s','z','j','h'," "],
	'rr': ['j','y','l','r','w'," "],
	'wr': ['j','y','l','r','w'," "],
	'rh': ['j','y','l','r','w'," "],
	'ss': ['f','v','s','z','j','h'," "],
	'sc': ['f','v','s','z','j','h'," "],
	'tt': ['p','b','t','d','k','g'," "],
	'th': ['f','v','s','z','j','h'," "],
	'zz': ['f','v','s','z','j','h'," "],
    'ai': ['i','e','a'," "],
    'ea': ['ay','ae','ei','eo','a'," "],
    'ui': ['i'," "],
    'oo': ['u','ou'," "],
    'ou': ['u','oo'," "],
    'eu': ['u', " "],
	}

def read_file(path: str) -> list:
    try:
        print("Attempting to read file...", end='')
        file = open(path, 'r', encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {path}")
    else:
        print("Success!")
        lines = file.readlines()
        file.close()
        return lines

def make_final(orig: list) -> None:
    output = []
    for i in range(len(orig)):
        output.append("")
    return output

def zip_sentences(orig: list, final: list) -> str:
    for i in range(len(final)):
        if final[i] == "":
            final[i] = orig[i]
    final = re.sub("\s+", " ", " ".join(final))
    return final

def get_next_available(orig: list, cost: int) -> list:
    if len("".join(orig)) == 0:
        return []
    else:
        gap = cost - 1
        random_indicies = random.sample(range(0,len(orig)),len(orig))
        for i in random_indicies:
            if cost > 1:
                if i + gap < len(orig) and orig[i + gap] != "" and orig[i] != "":
                    return [i,i+gap]
                elif i - gap >= 0 and orig[i - gap] != "" and orig[i] != "":
                    return [gap-i,i]
            if cost == 1:
                if orig[i] != "":
                    return [i]
        return []

def manner_swap(orig: list, final: list):
    """
    Swap two sounds that have similar manner of articulation.

    :params: A list of the original words and a list of the
    final words in process
    :returns: The updated original list and final list plus
    a Boolean where if orig and final lists were updated is
    equal to True, otherwise False
    """
    cost = 1
    index = get_next_available(orig, cost)
    if len(index) == 1:
        word= orig[index[0]]
        # two,too,to is an issue, so capture that here
        ttt = ['two','too','to']
        if word in ttt:
            # Only get a 'two' other than the one found
            ttt.remove(word)
            random_index0 = random.randint(0, len(ttt)-1)
            new_word = ttt[random_index0]
            # swap
            orig[index[0]] = ""
            final[index[0]] = new_word
            return orig, final, True
        elif len(word) == 1:
            # Handle words of length one such as 'a' or 'I' or abbreviations
            if word in MANNER_MAPPINGS:
                random_index1 = random.randint(0,len(MANNER_MAPPINGS[word])-1)
                new_word  = MANNER_MAPPINGS[word][random_index1]
                # swap
                orig[index[0]] = ""
                final[index[0]] = new_word
                return orig, final, True
            else:
                # Single word not in articulation dictionary
                return orig, final, False
        else:
            wordBigrams  = [[x[0],x[1]] for x in nltk.bigrams(word)]
            random_index2 = random.randint(0, len(wordBigrams)-1)
            # Get random individual sounds
            pair = wordBigrams[random_index2]
            # Also get them together in case that makes more sense('sh', 'ss', 'rr', etc.)
            combo = "".join(pair)
            if combo in MANNER_MAPPINGS:
                random_index3 = random.randint(0,len(MANNER_MAPPINGS[combo])-1)
                new_letters = MANNER_MAPPINGS[combo][random_index3]
                wordBigrams[random_index2] = ["", new_letters]
                if len(new_letters) == 2:
                    new_word = wordBigrams[0][0] + "".join([x[1] for x in wordBigrams[1:]])
                else:
                    new_word = wordBigrams[0][0] + "".join([x[1] for x in wordBigrams])
                # swap
                orig[index[0]] = ""
                final[index[0]] = new_word
                return orig,final, True
            else:
                letter = pair[1]
                if letter in MANNER_MAPPINGS:
                    random_index5 = random.randint(0,len(MANNER_MAPPINGS[letter])-1)
                    new_letter = MANNER_MAPPINGS[letter][random_index5]
                    pair[1] = new_letter
                    wordBigrams[random_index2] = pair
                    new_word = wordBigrams[0][0] + "".join([x[1] for x in wordBigrams])
                    # swap
                    orig[index[0]] = ""
                    final[index[0]] = new_word
                    return orig, final,True
                else:
                    return orig, final, False
    else: return orig, final, False



def find_homophone(orig: list, final: list):
    """
    Swap a word with it's homophone if it exists.

    :params: A list of the original words and a list of the
    final words in process
    :returns: The updated original list and final list plus
    a Boolean where if orig and final lists were updated is
    equal to True, otherwise False
    """
    acceptable_characters = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'}
    cost = 1
    index = get_next_available(orig, cost)
    if len(index) == 1:
        word = orig[index[0]]
        # if contains unusual characters, try a different method
        if len(set(word.strip()).difference(acceptable_characters)) > 0:
            return orig, final, False
        word = Homophones(word)
        # Find all possible homophones for given word
        possible_homophones = word.find_homophones()
        if possible_homophones is not None and 'no homophones for' not in possible_homophones and len(possible_homophones) != 0:
            results = [re.split(" is a homophone of ", x)[1] for x in possible_homophones]
            # Randomly select one of the homophones for swapping
            rand_index = random.randint(0,len(results)-1)
            homophone = results[rand_index]
            # Swap
            final[index[0]] = homophone
            orig[index[0]] = ""
            return orig, final, True
        else:
			# No available word homophone for homophone swapping (aka swap unsuccessful)
            return orig, final, False
    else:
        # No available word in orig for homophone swapping (aka swap unsuccessful)
        return orig, final, False

def assimilation(orig: list, final: list):
    """
    Cheap version of assimilation. Assimilate two words
    together by removing the first sound of the second
    word then creating a new word by merging the first
    word and second edited word together.

    :params: A list of the original words and a list of the
    final words in process
    :returns: The updated original list and final list plus
    a Boolean where if orig and final lists were updated is
    equal to True, otherwise False
    """
    cost = 2
    indexes = get_next_available(orig, cost)
    if len(indexes) == 2:
        word1 = orig[indexes[0]]
        word2 = orig[indexes[1]]
        # Merge together
        word3 = word1[:-1] + word2
        # Replace slot in final
        final[indexes[0]] = word3
        # Replace original with "" in orig
        orig[indexes[0]] = ""
        orig[indexes[1]] = ""
        return orig, final, True
    else:
        # Did not do assimilation successfully since 
        # not 2 consecutive words available in orig
        return orig, final, False

def do_noising(orig, final, nType):
    """
    Noising Control.

    :params: A list of the original words and a list of the
    final words in process
    :returns: Return result of noising if successful, or
    return original and final lists unaltered with a False
    boolean if noising type is not listed in the conditional
    options.
    """
    if nType == "assimilation":
        return assimilation(orig, final)
    elif nType == "homophone":
        return find_homophone(orig, final)
    elif nType == "manner":
        return manner_swap(orig, final)
    else:
        return orig, final, False


def noise(limit: int, orig: list, final: list, guidebook: dict) -> str:
    """
    Core noising function. Uses recursion to continually noise the given
    sentence until the limit is reduced to zero. Each noising action has
    an associated cost that is subtracted from the limit if successful.
    We know if noising is successful based on the Boolean value returned
    by each individual noising function.

    :params: limit - how many noising operations to perform, orig - the
    original sentence, final - an array of empty strings the same size as
    the orig list, guidebook - a dictionary of noising types and their costs
    :returns: The zipped lists of the original sentence (which should be
    full of empty strings if noising went well) and the newly noised
    sentence
    """
    if limit <= 0:
        return zip_sentences(orig, final)
    else:
        # Randomly get type of noising from options so that there is
        # no noising bias
        noise_type, cost = random.choice(list(guidebook.items()))
        if (limit - cost) >= 0:
            orig, final, status = do_noising(orig, final, noise_type)
            # Status makes sure the noising was successful; if it wasn't this
            # allows the sentence to be noised in some other way and thus keeps the
            # percent noised accurate
            if status is True:
                #print(f"Type {noise_type} imposed on sentence.")
                limit = limit - cost
            return noise(limit, orig, final, guidebook)
        else:
            return noise(limit, orig, final, guidebook)


def control(lines: list, percent: float, guidebook: dict):
    """
    Reads in sentences, performs noising on each sentence according
    to the noising percentage given.

    :params: lines - list of all sentences, percent - float of percent
    to be noised per sentence, guidebook - a dictionary of noising types
    and their costs
    """
    orig_noised = []
    start = time.time()
    count = 0
    for line in tqdm(lines):
        line = re.sub(r'\s+', ' ', line).rstrip()
        # do not include empty lines (\n, \t, '', \s, etc)
        if len(re.findall(r'^[\n\s\t]*$', line)) != 0:
            continue
        # tokenize
        orig = line.split(" ")
        # get limit
        limit = float(round(percent*len(orig)))
        # get final list
        final = make_final(orig)
        if limit < 1:
            # Too small to do more than one noising action
            small_orig, small_final, boolean = manner_swap(orig, final)
            if boolean:
                noised_sent = zip_sentences(small_orig, small_final)
            else:
                # Too small to be noised so in this case
                # orig == noised_orig
                noised_sent = orig
        else:
            # normal noise
            noised_sent = noise(limit, orig, final, guidebook)
        # add as tuple to noised (they are strings again at this point)
        orig_noised.append([noised_sent, line.rstrip()])
        #print(f"ORIG: {line}\nNOISED: {noised_sent}")
        count += 1
    print(f"Total time noising: {time.time() - start}")
    print(f"Total sentences noised: {count}")
    return orig_noised

def to_csv(sentences: list, output_path: str, file_name: str) -> None:
    """
    Takes a list of noised and clean sentences and writes them to
    a csv file.

    :params: sentences - list of all sentences, output_path - directory
    to which the csv file will be written, file_name - name of file to
    be written
    :returns: None
    """
    path      = os.path.join(output_path, file_name)
    # open file to write 
    file      = open(path, "w", encoding="utf-8", newline='')
    # create instance of csv writer
    writer    = csv.writer(file)
    # headers
    headers = ['befr', 'en']
    # write headers first
    writer.writerow(headers)
    # write data to file
    writer.writerows(sentences)
    print(f"Noised data writtent to CSV file at {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",         required=True,   help="Path to txt file containing transcripts")
    parser.add_argument("--percent",      required=True,   help="Percent of data to be noised")
    parser.add_argument("--outpath",      required=True,   help="Path to which to write noised/unnoised sentence pair csv")
    parser.add_argument("--output-name",  required=True,   help="Name for new noised csv file")
    args = parser.parse_args()
    
    file_lines   = read_file(args.path)

    # Weights for each noising type
    guidelines = {
                'assimilation' : 2.0,
                'homophone'    : 1.0,
                'manner'       : 0.5,
            }

    output  = control(file_lines, float(args.percent), guidelines)
    outPath = os.path.join(args.outpath,"")
    name    = f"NOISED-{args.percent}_{args.output_name}.csv"
    to_csv(output, outPath, name)


if __name__ == "__main__":
    main()



