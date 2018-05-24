import os

import argparse

from opinions.scripts.utils import load_json, get_conll


def load_data(path):
    conll = get_conll(os.path.join(path, "train.conll"))
    skladnica, opta = load_json(os.path.join(path, "skladnica_output.json"), os.path.join(path, "OPTA-treebank-0.1.json"))
    return conll, opta, skladnica


def write_sentences(conll_sent, opta_sent, skladnica_sent):
    with open("resources/conll_sent.txt", "w") as f:
        f.write("\n".join(conll_sent))
    with open("resources/opta_sent.txt", "w") as f:
        f.write("\n".join(opta_sent))
    with open("resources/skladnica_sent.txt", "w") as f:
        f.write("\n".join(skladnica_sent))


def remove_duplicates(conll, opta, skladnica):
    conll_sent = [" ".join([token.strip().split(" ")[1] for token in sentence]) for sentence in conll]
    opta_sent = [" ".join([word.strip().split("\t")[1] for word in sentence['parsedSent']]) for sentence in opta]
    skladnica_sent = [" ".join([word.strip().split("\t")[1] for word in sentence['parsedSent']]) for sentence in skladnica]

    skladnica_without_duplicates = []
    skladnica_sents_removed = 0
    for sentence in skladnica_sent:
        if sentence not in conll_sent or sentence not in opta_sent:
            skladnica_without_duplicates.append(sentence)
        else:
            skladnica_sents_removed += 1

    conll_without_duplicates = []
    conll_sents_removed = 0
    for sentence in conll_sent:
        if sentence not in opta_sent:
            conll_without_duplicates.append(sentence)
        else:
            conll_sents_removed += 1

    print("Removed {0} sentences out of {1} from skladnica_output.json file."
          .format(str(skladnica_sents_removed), str(len(skladnica_sent))))
    print("Removed {0} sentences out of {1} from train.conll file."
          .format(str(conll_sents_removed), str(len(conll_sent))))

    return conll_without_duplicates, opta_sent, skladnica_without_duplicates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Removes duplicates from corpuses and converts them to .conll')
    parser.add_argument('path', type=str, help='Path to directory with corpuses')
    args = parser.parse_args()

    conll, opta, skladnica = load_data(args.path)
    write_sentences(*remove_duplicates(conll, opta, skladnica))
