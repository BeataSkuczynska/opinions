import argparse
import json
from os import path
from random import shuffle

from sklearn.model_selection import train_test_split


def get_data(path):
    with open(path) as f:
        data = json.load(f)
    shuffle(data)
    parsed = []
    for sentence in data:
        sentence = sentence['parsedSent']
        parsed.append(sentence)
    print (len(parsed))
    return parsed


def save_train_test_conlls(parsed_train, parsed_test):
    basepath = path.dirname(__file__)
    # train_path = path.abspath(path.join(basepath, "..", "opinions/opta-tagger/train_data/conll-format/skladnica.txt"))
    with open("resources/S_skladnica.conll", "w") as f:
        for parsed_sentence in parsed_train:
            for parsed_word in parsed_sentence:
                parsed_word = parsed_word.replace("\n", "")
                f.write(parsed_word + "\n")
            f.write("\n")
        # parsed_train = ["\n".join(parsed_sentence) for parsed_sentence in parsed_train]
        # parsed_train = "\n".join(parsed_train)
        # f.write(parsed_train)
    with open('resources/input.conll', "w") as f:
        for parsed_sentence in parsed_test:
            for parsed_word in parsed_sentence:
                parsed_word = parsed_word.strip().replace("\n", "")
                tab_split_word = parsed_word.split("\t")
                tab_split_word[-1] = "_"
                parsed_word = "\t".join(tab_split_word)
                f.write(parsed_word + "\n")
            f.write("\n")
        # parsed_test = ["\n".join(parsed_sentence) for parsed_sentence in parsed_test]
        # parsed_test = "\n".join(parsed_test)
        # f.write(parsed_test)


def save_parsed(parsed):
    with open("resources/S_skladnica.conll", "w") as f:
        for parsed_sentence in parsed:
            for parsed_word in parsed_sentence:
                parsed_word = parsed_word.replace("\n", "")
                tab_split_word = parsed_word.split("\t")
                tab_split_word.insert(-2, "_")
                parsed_word = " ".join(tab_split_word)
                f.write(parsed_word + "\n")
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for CRF training')
    parser.add_argument('path', type=str, help='Path to OPTA/skladnica in JSON format')
    args = parser.parse_args()

    parsed = get_data(args.path)
    parsed_train, parsed_test = train_test_split(parsed)
    save_train_test_conlls(parsed_train, parsed_test)
    # save_parsed(parsed)
