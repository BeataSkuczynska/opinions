import argparse

from opinions.utils import load_json, save_conll


def format_test_set(data):
    conll_format_sentences = []
    for entry in data:
        sentence = entry['parsedSent']
        sentence = [word.strip().replace("\n", "").replace("\t", " ") for word in sentence]
        sentence_formatted = []
        for token in sentence:
            token_splitted = token.split(" ")
            if token_splitted[-1] in ["S", "T"]:
                token_splitted[-1] = "_"
            sentence_formatted.append(" ".join(token_splitted))
        conll_format_sentences.append(sentence_formatted)
    return conll_format_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for CRF training')
    parser.add_argument('path', type=str, help='Path to Skladnica treebank in JSON format')
    args = parser.parse_args()

    data = load_json(args.path)
    formatted_sents = format_test_set(data)
    save_conll(formatted_sents, "skladnica_test")
