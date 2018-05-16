import argparse
import json
import os
from random import shuffle

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
from keras import Input, Model, optimizers
from keras.layers import Flatten, Bidirectional, Concatenate, Reshape, GRU
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

max_len = 20

def parse_data(path):
    with open(path) as f:
        data = json.load(f)
    shuffle(data)
    word2index = dict()
    i = 0
    parsed = []
    sentiments = []
    targets = []
    for sentence in data:
        sentence = sentence['parsedSent']
        new_sentence = []
        sentiment = []
        target = []
        # max_len = max(max_len, len(sentence))
        idx = 0
        added = 0
        for x in sentence:
            if idx ==max_len:
                break
            idx+=1
            y = x.split("\t")
            if y[-2] == 'S':
                sentiment.append(1)
            else:
                sentiment.append(0)
            if y[-2] == 'T' or  y[-2] == 'A':
                added = 1
                target.append(1)
            else:
                target.append(0)
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


def load_embeddings(embeddings_path):
    if os.path.isfile(embeddings_path + '.model'):
        model = KeyedVectors.load(embeddings_path + ".model")
    if os.path.isfile(embeddings_path + '.vec'):
        model = FastText.load_word2vec_format(embeddings_path + '.vec')
    return model


def create_model(embeddings, max_features=50, maxlen=45, ):
    sentiments_input = Input(shape=(maxlen,), name='sentiments')
    sentiments = Reshape((maxlen, 1,))(sentiments_input)
    emb_input = Input(shape=(maxlen,), name='emb_input')
    embedding = Embedding(embeddings.shape[0], max_features, input_length=maxlen, weights=[embeddings], trainable=True)(
        emb_input)
    concat = Concatenate(axis=-1)([embedding, sentiments])
    lstm = Bidirectional(GRU(output_dim=200, activation='relu', return_sequences=True))(concat)
    dropout = Dropout(0.5)(lstm)
    dense = Dense(1, activation='sigmoid')(dropout)
    out = Flatten()(dense)

    model = Model(inputs=[emb_input, sentiments_input], outputs=[out])
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


def train(path, emb):
    sentences, sentiments, targets, max_len, word2index = parse_data(path)
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
    parsed_train, parsed_test, targets_train, targets_test, sentiments_train, sentiments_test = train_test_split(sentences,
                                                                                                                 targets,
                                                                                                                 sentiments,
                                                                                                                 test_size=0.10,
                                                                                                                 random_state=42)

    targets_train = pad_sequences(targets_train, maxlen=max_len)
    padded_targets_test = pad_sequences(targets_test, maxlen=max_len)

    model = create_model(index2vec, max_features=50, maxlen=max_len)
    model.fit([parsed_train, sentiments_train], targets_train, batch_size=16, nb_epoch=100, validation_split=0.2,
              verbose=1)

    loss, accuracy = model.evaluate([parsed_train, sentiments_train], targets_train, verbose=0)
    print('Accuracy train: %f' % (accuracy * 100))

    loss, accuracy = model.evaluate([parsed_test, sentiments_test], padded_targets_test, verbose=0)
    print('Accuracy test: %f' % (accuracy * 100))

    predictions =np.around(model.predict([parsed_test, sentiments_test], verbose=0)).astype('int')
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
    args = parser.parse_args()
    train(args.path, args.emb)
