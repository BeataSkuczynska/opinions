import json
from random import shuffle


def parse_data(path):
    with open(path) as f:
        data = json.load(f)
    shuffle(data)
    word2index = dict()
    i = 0
    max_len = 0
    parsed = []
    sentiments = []
    targets = []
    for sentence in data:
        sentence = sentence['parsedSent']
        new_sentence = []
        sentiment = []
        target = []
        max_len = max(max_len, len(sentence))
        idx = 0
        added = 0

        for x in sentence:
            if idx ==max_len:
                break
            idx+=1
            y = x.split("\t")
            if y[-2] == 'S':
                sentiment.append(2)
            else:
                sentiment.append(0)
            if y[-2] == 'T' or y[-2] == 'A':
                added = 1
                target.append(2)
            else:
                target.append(1)
            word = y[2]

            if word not in word2index:
                word2index[word] = i
                i += 1
            new_sentence.append(word2index[word])
        if added ==1:
            parsed.append(new_sentence)
            sentiments.append(sentiment)
            targets.append(target)
    return parsed, sentiments, targets, max_len, word2index
