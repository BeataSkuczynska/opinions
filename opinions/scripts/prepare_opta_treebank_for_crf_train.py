import argparse
import os

from opinions.scripts.utils import load_json, save_conll


def format_data(data):
    conll_format_sentences = []
    for entry in data:
        sentence = entry['parsedSent']
        rule_id = entry['rule_id']
        isAtrRelToSent = entry['isAtrRelToSent']
        sentence = [word.strip().replace("\t", " ") + " _ _ _ _" for word in sentence]
        sentence_formatted = []
        for token in sentence:
            token_splitted = token.split(" ")
            if token_splitted[-5] == "A":
                token_splitted[-5:-1] = ["_", "_", "R", str(rule_id)]
                if bool(isAtrRelToSent):
                    token_splitted[-1] = "A"
            elif token_splitted[-5] == "S":
                token_splitted[-5:-3] = ["_", "S"]
            sentence_formatted.append(" ".join(token_splitted))
        conll_format_sentences.append(sentence_formatted)
    return conll_format_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert .json file to conll and prepare it for training of opta-tagger')
    parser.add_argument('input_path', help='Path to OPTA treebank in JSON format', type=str)
    parser.add_argument('output_path', help='Path to directory to write conll file', type=str)
    args = parser.parse_args()

    data = load_json(args.input_path)
    formatted_sents = format_data(data)
    save_conll(formatted_sents, os.path.join(args.outputh_path, "opta_train"))

