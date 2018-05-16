import argparse
import os

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from opinions.dataset import parse_data
from opinions.model import create_model


def load_embeddings(embeddings_path):
    if os.path.isfile(embeddings_path + '.model'):
        model = KeyedVectors.load(embeddings_path + ".model")
    if os.path.isfile(embeddings_path + '.vec'):
        model = FastText.load_word2vec_format(embeddings_path + '.vec')
    return model


def train(path, emb, max_len):
    sentences, sentiments, targets, max_len, word2index = parse_data(path, max_len)
    emb = load_embeddings(emb)
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
    parsed_train, parsed_test, targets_train, targets_test, sentiments_train, sentiments_test = train_test_split(
        sentences,
        targets,
        sentiments,
        test_size=0.10,
        random_state=42)

    targets_train = pad_sequences(targets_train, maxlen=max_len)
    padded_targets_test = pad_sequences(targets_test, maxlen=max_len)

    model = create_model(index2vec, max_features=50, maxlen=max_len)
    model.fit([parsed_train, sentiments_train], targets_train, batch_size=16, nb_epoch=30, validation_split=0.2,
              verbose=1)

    loss, accuracy = model.evaluate([parsed_train, sentiments_train], targets_train, verbose=0)
    print('Accuracy train: %f' % (accuracy * 100))

    loss, accuracy = model.evaluate([parsed_test, sentiments_test], padded_targets_test, verbose=0)
    print('Accuracy test: %f' % (accuracy * 100))

    predictions = np.around(model.predict([parsed_test, sentiments_test], verbose=0)).astype('int')
    unpad = []
    for idx, val in enumerate(targets_test):
        unpad.append(predictions[idx][0:len(val)].tolist())
    flat_predictions = [i for x in unpad for i in x]
    flat_target = [i for x in targets_test for i in x]
    confusion_m = confusion_matrix(flat_target, flat_predictions)
    print(confusion_m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for option NN training')
    parser.add_argument('path', type=str, help='Data training path')
    parser.add_argument('emb', type=str, help='Embedding file path')
    parser.add_argument('max_len', type=int, help='Max sentence length')
    args = parser.parse_args()
    train(args.path, args.emb, args.max_len)

