import argparse
import os

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from opinions.dataset import parse_data
from opinions.model import create_model
import opinions.config


def load_embeddings(embeddings_path):
    if os.path.isfile(embeddings_path + '.model'):
        model = KeyedVectors.load(embeddings_path + ".model")
    if os.path.isfile(embeddings_path + '.vec'):
        model = FastText.load_word2vec_format(embeddings_path + '.vec')
    return model


def prepare_data(path, emb):
    sentences, sentiments, targets, max_len, word2index = parse_data(path)
    unk_count = 0
    vocab_size = len(word2index)
    index2vec = np.zeros((vocab_size + 1, emb.vector_size), dtype="float32")
    index2vec[0] = np.zeros(emb.vector_size)
    for word in word2index:
        index = word2index[word]
        try:
            index2vec[index] = emb[word]
        except KeyError:
            index2vec[index] = np.random.rand(emb.vector_size)
            unk_count += 1

    print("emb vocab size: ", len(emb.vocab))
    print("unknown words count: ", unk_count)
    print("index2vec size: ", len(index2vec))
    print("words  ", len(word2index))

    sentences = pad_sequences(sentences, maxlen=max_len)
    sentiments = pad_sequences(sentiments, maxlen=max_len)
    mlb = OneHotEncoder()
    mlb.fit([[0], [1], [2]])
    parsed_train, parsed_test, targets_train, targets_test, sentiments_train, sentiments_test = train_test_split(
        sentences,
        targets,
        sentiments,
        test_size=0.10,
        shuffle=False)

    targets_train = pad_sequences(targets_train, maxlen=max_len)
    padded_targets_test = pad_sequences(targets_test, maxlen=max_len)

    targets_train = np.asarray([mlb.transform([[i] for i in x]).toarray() for x in targets_train])
    padded_targets_test = np.asarray([mlb.transform([[i] for i in x]).toarray() for x in padded_targets_test])
    return parsed_train, parsed_test, sentiments_train, sentiments_test, targets_train, targets_test, padded_targets_test, index2vec, max_len


def train_eval(values, emb, config=opinions.config.params):
    parsed_train, parsed_test, sentiments_train, sentiments_test, targets_train, targets_test, padded_targets_test, index2vec, max_len = values
    model = create_model(index2vec, max_features=emb.vector_size, maxlen=max_len, params=config)
    model.fit([parsed_train, sentiments_train], targets_train, batch_size=16, nb_epoch=10, validation_split=0.2,
              verbose=0)

    loss, train_accuracy = model.evaluate([parsed_train, sentiments_train], targets_train, verbose=0)
    print('Accuracy train: %f' % (train_accuracy * 100))

    loss, test_accuracy = model.evaluate([parsed_test, sentiments_test], padded_targets_test, verbose=0)
    print('Accuracy test: %f' % (test_accuracy * 100))

    predictions = np.around(model.predict([parsed_test, sentiments_test], verbose=0))
    unpad = []
    for idx, val in enumerate(targets_test):
        unpad.append(np.argmax(predictions[idx], axis = 1).tolist())
        # unpad.append(predictions[idx][0:len(val)].tolist())
    flat_predictions = [i for x in unpad for i in x]
    flat_target = [np.argmax(i) for x in padded_targets_test for i in x]
    confusion_m = confusion_matrix(flat_target, flat_predictions)
    print(confusion_m)
    return train_accuracy, test_accuracy, confusion_m


def train(path, emb, config=opinions.config.params):
    values = prepare_data(path, emb)
    return train_eval(values, emb,config)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Arguments for option NN training')
    # parser.add_argument('path', type=str, help='Data training path')
    # parser.add_argument('emb', type=str, help='Embedding file path')
    # parser.add_argument('max_len', type=int, help='Max sentence length')
    # args = parser.parse_args()
    # emb = load_embeddings(args.emb)
    # train(args.path, emb, args.max_len)

    # train("/home/komputerka/opinion_target/json/OPTA-treebank-0.1.json",
    #       load_embeddings("/home/komputerka/opinion_target/w2v_allwiki_nkjp300_50"), 20)
    train("json/OPTA-treebank-0.1.json", load_embeddings("w2v_allwiki_nkjp300_50"))
