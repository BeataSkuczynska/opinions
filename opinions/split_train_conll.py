import argparse

from sklearn.model_selection import train_test_split

from opinions.utils import get_conll


def save_train_test_conlls(parsed_train, parsed_test):
    with open("resources/traintrain.conll", "w") as f:
        for parsed_sentence in parsed_train:
            for parsed_word in parsed_sentence:
                f.write(parsed_word)
            f.write("\n")
    with open('resources/traintest.conll', "w") as f:
        for parsed_sentence in parsed_test:
            for parsed_word in parsed_sentence:
                parsed_word = parsed_word.strip()
                split_word = parsed_word.split(" ")[:-7]
                split_word.extend(["_", "_", "_"])
                parsed_word = "\t".join(split_word)
                f.write(parsed_word + "\n")
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for CRF training')
    parser.add_argument('path', type=str, help='Path to train CONLL file')
    args = parser.parse_args()

    parsed = get_conll(args.path)
    parsed_train, parsed_test = train_test_split(parsed, test_size=0.1)
    save_train_test_conlls(parsed_train, parsed_test)
