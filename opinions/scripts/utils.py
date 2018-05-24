import json


def load_json(*paths):
    data = []
    for path in paths:
        with open(path) as f:
            data.append(json.load(f))
    return data


def get_conll(path):
    with open(path) as f:
        parsed_all = []
        parsed_sent = []
        for line in f.readlines():
            if len(line) > 2:
                parsed_sent.append(line)
            else:
                parsed_all.append(parsed_sent)
                parsed_sent = []
    return parsed_all


def save_conll(formatted_sents, output_path):
    with open(output_path, "w") as f:
        for sentence in formatted_sents:
            for word in sentence:
                f.write(word + "\n")
            f.write("\n")
