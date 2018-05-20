import json
import re



def get_sentiment(row, sentiment):
    if row[-2] == 'S' or row[-4] == 'S' or row[-3] == 'S':
        sentiment.append(2)
    else:
        sentiment.append(1)
    return sentiment


def get_target(row, target, added):
    if row[-2] == 'A' or row[-1] == 'A'  or row[-3] == 'A'or row[-2] == 'T' or row[-1] == 'T'  or row[-3] == 'T':
        added = 1
        target.append(2)
    else:
        target.append(1)
    return target, added


def parse_data(paths, vocab = None, max_len = None):
    raw = []
    relevants = []
    for path in paths:
        corpus = []
        with open(path) as f:
            if path.endswith(".json"):
                data = json.load(f)
                corpus = [sentence['parsedSent'] for sentence in data]
                relevant = [sentence['isAtrRelToSent'] for sentence in data]
            elif path.endswith(".conll"):
                sentence = []
                for line in f.read().splitlines():
                    if line:
                        sentence.append(line)
                    else:
                        corpus.append(sentence)
                        sentence = []
            else:
                raise Exception("Unknown file type")
            relevants.extend(relevant)
            raw.extend(corpus)
    word2index = dict() if vocab is None else vocab
    i = 0

    max_len_count = 0
    parsed = []
    sentiments = []
    targets = []
    for sentence in raw:
        new_sentence = []
        sentiment = []
        target = []
        max_len_count = max(max_len_count, len(sentence))
        idx = 0
        added = 0
        for word in sentence:
            idx += 1
            row = re.split("\s", word)
            sentiment = get_sentiment(row, sentiment)
            target, added = get_target(row, target, added)
            word = row[2]


            if word not in word2index:
                word2index[word] = i
                i += 1
            new_sentence.append(word2index[word])

        parsed.append(new_sentence)
        sentiments.append(sentiment)
        targets.append(target)
    if not vocab:
        with open("vocab.json", "w+") as f:
            json.dump(word2index, f)
    if max_len is None:
        max_len = max_len_count
    open("max_len", "w+").write(str(max_len))

    return parsed, sentiments, targets, max_len, word2index, relevants
