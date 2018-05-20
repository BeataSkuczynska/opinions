import json
import pickle

import numpy as np

from keras.models import load_model

from train import prepare_data, load_embeddings


def predict(path, emb,vocab, max_len, index2vec, model = None):
    # model = load_model('model.h5')
    _, sentences, _, sentiments, _, targets, padded_targets, _, max_len, _, relevant = prepare_data([path], emb, test=1.0, vocab=vocab,
                                                                                       max_len=max_len, index2vec = index2vec)
    predictions = model.predict([sentences, sentiments, relevant], verbose=0)
    print(model.evaluate([sentences, sentiments, relevant], padded_targets))
    unpad = ""
    for idx, val in enumerate(targets):
        for prediction in predictions[idx][max_len - len(val):].tolist():
            predict_val = np.argmax(prediction)
            if predict_val != 2:
                unpad += "_"
            else:
                unpad += "A"
            unpad += "\n"
        unpad += "\n"
    with open("rnn_output.conll", "w+") as f:
        f.write(unpad)


